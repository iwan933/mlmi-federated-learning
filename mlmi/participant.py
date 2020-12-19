from pathlib import Path
from typing import Dict, List, Union

import torch
from torch import Tensor
from torch.utils import data

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks.base import Callback

from mlmi.struct import ExperimentContext, TrainArgs, ModelArgs
from mlmi.log import getLogger
from mlmi.settings import CHECKPOINT_DIR


logger = getLogger(__name__)


class BaseParticipant(object):

    def __init__(self, participant_name: str, model_args: ModelArgs, context: ExperimentContext):
        assert participant_name is not None, 'A participant name is required to load and save logs'
        assert model_args is not None, 'Model args are required to initialize a model for the participant'
        assert context is not None, 'Experiment context is required for participant'

        self._name = participant_name
        self._cluster_id = None
        self.experiment_context = context
        self.model_args = model_args
        self.model = model_args.model_class(*model_args.args, **model_args.kwargs)

    def get_model(self) -> pl.LightningModule:
        """
        The model to train
        :return: The model
        """
        return self.model

    def load_model_state(self, model_state: Dict[str, Tensor]):
        """
        Loads the model state into the current model instance
        :param model_state: The model state to load
        """
        self.model.load_state_dict(model_state)

    def load_model_state_from_checkpoint(self):
        """
        Load the model state from an existing saved checkpoint
        :param checkpoint_path: Path to the checkpoint file
        """
        self.model = self.model_args.model_class.load_from_checkpoint(
            checkpoint_path=str(self.get_checkpoint_path().absolute()))

    def get_checkpoint_path(self, suffix: Union[str, None] = None) -> Path:
        """
        Constructs a checkpoint path based on
        :return:
        """
        _suffix = '' if suffix is None else '_' + suffix
        filename = (self._name + _suffix + '.ckpt')
        return CHECKPOINT_DIR / self.experiment_context.name / filename

    def save_model_state(self):
        """
        Saves the model state of the aggregated model
        :param target_path: The path to save the model at
        :return:
        """
        torch.save(self.model.state_dict(), self.get_checkpoint_path())

    @property
    def cluster_id(self) -> str:
        return self._cluster_id

    @cluster_id.setter
    def cluster_id(self, value: str):
        self._cluster_id = value


class BaseTrainingParticipant(BaseParticipant):
    def __init__(self, client_id: str, model_args: ModelArgs, context: ExperimentContext,
                 train_dataloader: data.DataLoader, num_train_samples: int,
                 test_dataloader: data.DataLoader, num_test_samples: int,
                 lightning_logger: LightningLoggerBase, *args, **kwargs):
        super().__init__(client_id, model_args, context)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.lightning_logger = lightning_logger
        self.callbacks = None
        self.model_state = None

    def get_logger(self) -> LightningLoggerBase:
        """
        Gets the logger to use for the training in later stage.
        :return: The lightning logger to use
        """
        return self.lightning_logger

    def get_trainer(self, **kwargs) -> pl.Trainer:
        """
        Creates a new trainer instance for each training round.
        :param kwargs: additional keyword arguments to send to the trainer for configuration
        :return: a pytorch lightning trainer instance
        """
        return pl.Trainer(callbacks=self.callbacks, limit_val_batches=0.0, logger=self.lightning_logger,
                          **kwargs)

    def set_trainer_callbacks(self, callbacks: List[Callback]):
        self.callbacks = callbacks

    def get_train_data_loader(self) -> data.DataLoader:
        return self.train_dataloader

    def get_test_data_loader(self) -> data.DataLoader:
        return self.test_dataloader

    def get_num_train_samples(self) -> int:
        return self.num_train_samples

    def get_num_test_samples(self) -> int:
        return self.num_test_samples

    def train(self, training_args: TrainArgs, *args, **kwargs):
        """
        Implement the training routine.
        :param training_args:
        :param args:
        :param kwargs:
        :return:
        """
        trainer = self.get_trainer(**training_args.kwargs)
        local_model = self.get_model()
        train_dataloader = self.get_train_data_loader()
        trainer.fit(local_model, train_dataloader, train_dataloader)
        self.save_model_state()


class BaseAggregatorParticipant(BaseParticipant):

    def __init__(self, participant_name: str, model_args: ModelArgs, context: ExperimentContext):
        super().__init__(participant_name, model_args, context)

    def aggregate(self, participants: List[BaseParticipant], *args, **kwargs):
        """
        Aggregate the models of other participants with their models.
        :param participants: Participants to apply the model changes from
        :return:
        """
        raise NotImplementedError()
