from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor, optim
from torch.utils import data

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks.base import Callback

from mlmi.structs import TrainArgs, ModelArgs
from mlmi.log import getLogger
from mlmi.settings import CHECKPOINT_DIR


logger = getLogger(__name__)


class BaseParticipant(object):

    def __init__(self, participant_name: str, model_args: ModelArgs, context):
        assert participant_name is not None, 'A participant name is required to load and save logs'
        assert model_args is not None, 'Model args are required to initialize a model for the participant'
        assert context is not None, 'Experiment context is required for participant'

        self._name = participant_name
        self._cluster_id = None
        self._experiment_context = context
        self._model_args = model_args
        self._model = model_args.model_class(*model_args.args, **model_args.kwargs, participant_name=participant_name)

    @property
    def model(self) -> pl.LightningModule:
        """
        The model to train
        :return: The model
        """
        return self._model

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, value: str):
        self._cluster_id = value

    def overwrite_model_state(self, model_state: Dict[str, Tensor]):
        """
        Loads the model state into the current model instance
        :param model_state: The model state to load
        """
        self._model.load_state_dict(model_state)
        self.save_model_state()

    def overwrite_optimizer_state(self, optimizer_state: Dict[str, Tensor]):
        """
        Overwrites the optimizer state
        :param optimizer_state: state to write
        :return:
        """
        self.model.optimizer.load_state_dict(optimizer_state)

    def load_model_state_from_checkpoint(self):
        """
        Load the model state from an existing saved checkpoint
        """
        self._model = self._model_args.model_class.load_from_checkpoint(
            checkpoint_path=str(self.get_checkpoint_path().absolute()))

    def get_checkpoint_path(self, suffix: Union[str, None] = None) -> Path:
        """
        Constructs a checkpoint path based on
        :return:
        """
        str_suffix = '' if suffix is None else '_' + suffix
        filename = (self._name + str_suffix + '.ckpt')
        return CHECKPOINT_DIR / self._experiment_context.name / filename

    def save_model_state(self):
        """
        Saves the model state of the aggregated model
        :param target_path: The path to save the model at
        :return:
        """
        path = self.get_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), path)


class BaseTrainingParticipant(BaseParticipant):
    def __init__(self, client_id: str, model_args: ModelArgs, context,
                 train_dataloader: data.DataLoader, num_train_samples: int,
                 test_dataloader: data.DataLoader, num_test_samples: int,
                 lightning_logger: LightningLoggerBase, *args, **kwargs):
        super().__init__(client_id, model_args, context)
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader
        self._num_train_samples = num_train_samples
        self._num_test_samples = num_test_samples
        self._lightning_logger = lightning_logger
        self._callbacks = None
        self._model_state = None
        self._trainer = None

    def create_trainer(self, enable_logging=True, **kwargs) -> pl.Trainer:
        """
        Creates a new trainer instance for each training round.
        :param kwargs: additional keyword arguments to send to the trainer for configuration
        :return: a pytorch lightning trainer instance
        """
        _kwargs = kwargs.copy()
        if enable_logging:
            _kwargs['logger'] = self.logger
        else:
            _kwargs['checkpoint_callback'] = False
            _kwargs['logger'] = False
        if torch.cuda.is_available():
            _kwargs['gpus'] = 1
        return pl.Trainer(callbacks=self._callbacks, limit_val_batches=0.0, **_kwargs)

    def set_trainer_callbacks(self, callbacks: List[Callback]):
        self._callbacks = callbacks

    @property
    def logger(self) -> LightningLoggerBase:
        """
        Gets the logger to use for the training in later stage.
        :return: The lightning logger to use
        """
        return self._lightning_logger

    @property
    def train_data_loader(self) -> data.DataLoader:
        return self._train_dataloader

    @property
    def test_data_loader(self) -> data.DataLoader:
        return self._test_dataloader

    @property
    def num_train_samples(self) -> int:
        return self._num_train_samples

    @property
    def num_test_samples(self) -> int:
        return self._num_test_samples

    def train(self, training_args: TrainArgs, *args, **kwargs):
        """
        Implement the training routine.
        :param training_args:
        :param args:
        :param kwargs:
        :return:
        """
        trainer = self.create_trainer(enable_logging=False, **training_args.kwargs)
        train_dataloader = self.train_data_loader
        trainer.fit(self.model, train_dataloader, train_dataloader)
        self.save_model_state()

    def test(self, model: Optional[torch.nn.Module] = None, use_local_model: bool = False):
        """
        Test the model state on this clients data.
        :param
        :param model_state: The model state to evaluate
        :return: The output loss
        """
        assert use_local_model or model is not None

        trainer = self.create_trainer(enable_logging=False)

        if use_local_model:
            model = self.model

        result = trainer.test(model=model, test_dataloaders=self.test_data_loader, verbose=False)
        return result


class BaseAggregatorParticipant(BaseParticipant):

    def __init__(self, participant_name: str, model_args: ModelArgs, context):
        super().__init__(participant_name, model_args, context)

    def aggregate(self, participants: List[BaseParticipant], *args, **kwargs):
        """
        Aggregate the models of other participants with their models.
        :param participants: Participants to apply the model changes from
        :return:
        """
        raise NotImplementedError()


class BaseParticipantModel(object):

    def __init__(self, *args, **kwargs):
        assert 'participant_name' in kwargs, 'Please provide a participant name parameter in model args to identify' \
                                             'your model in logging'
        self.participant_name = kwargs.pop('participant_name', None)
        super().__init__(*args, **kwargs)

    @property
    def optimizer(self) -> optim.Optimizer:
        raise NotImplementedError()
