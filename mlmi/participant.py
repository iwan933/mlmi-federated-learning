from typing import Dict, List

import torch
from torch import Tensor
from torch.utils import data

import pytorch_lightning as pl
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks.base import Callback

from mlmi.struct import TrainArgs, ModelArgs
from mlmi.log import getLogger


logger = getLogger(__name__)


class BaseParticipant(object):

    def __init__(self, model_args: ModelArgs):
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

    def load_model_state_from_checkpoint(self, checkpoint_path: str):
        """
        Load the model state from an existing saved checkpoint
        :param checkpoint_path: Path to the checkpoint file
        """
        self.model = self.model_args.model_class.load_from_checkpoint(checkpoint_path=checkpoint_path)

    def save_model_state(self, target_path: str):
        """
        Saves the model state of the aggregated model
        :param target_path: The path to save the model at
        :return:
        """
        torch.save(self.model.state_dict(), target_path)


class BaseTrainingParticipant(BaseParticipant):
    def __init__(self, client_id: str, model_args: ModelArgs, train_dataloader: data.DataLoader, num_train_samples: int,
                 test_dataloader: data.DataLoader, num_test_samples: int,
                 lightning_logger: LightningLoggerBase, callbacks: List[Callback] = None, *args, **kwargs):
        super().__init__(model_args)
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.lightning_logger = lightning_logger
        self.callbacks = callbacks
        self.model_state = None

    def get_id(self) -> str:
        """
        Participant id to identify the Participant
        :return: Participant id string
        """
        return self.client_id

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
        trainer = self.get_trainer(max_epochs=training_args.epochs,
                                   resume_from_checkpoint=training_args.resume_from_checkpoint,
                                   **training_args.kwargs)
        local_model = self.get_model()
        train_dataloader = self.get_train_data_loader()
        trainer.fit(local_model, train_dataloader, train_dataloader)


class BaseAggregatorParticipant(BaseParticipant):

    def __init__(self, aggregator_id: str, model_args: ModelArgs):
        super().__init__(model_args)
        self.id = aggregator_id

    def get_id(self):
        return self.id

    def aggregate(self, participants: List[BaseParticipant], *args, **kwargs):
        """
        Aggregate the models of other participants with their models.
        :param participants: Participants to apply the model changes from
        :return:
        """
        raise NotImplementedError()
