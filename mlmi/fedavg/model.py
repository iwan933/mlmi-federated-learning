from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from fedml_api.model.cv.cnn import CNN_DropOut

from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.struct import OptimizerArgs


logger = getLogger(__name__)


def add_weighted_model(previous: Optional[Dict[str, Tensor]], next: Dict[str, Tensor], num_samples: int,
                       num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = dict() if previous is None else previous
    for key, w in next.items():
        weighted_parameter = (num_samples / num_total_samples) * w
        if previous is None:
            weighted_model_state[key] = weighted_parameter
        else:
            weighted_model_state[key] = weighted_model_state[key] + weighted_parameter
    return weighted_model_state


def load_participant_model_state(participant: BaseParticipant) -> Dict[str, Tensor]:
    """
    Method to load the current model state from a participant
    :param participant: participant to load the model from
    :return:
    """
    return participant.model.state_dict()


class FedAvgClient(BaseTrainingParticipant):
    pass


class FedAvgServer(BaseAggregatorParticipant):
    total_train_sample_num = 0

    def get_total_train_sample_num(self):
        return self.total_train_sample_num

    def aggregate(self, participants: List[BaseParticipant], *args, **kwargs):
        num_train_samples: List[int] = kwargs.pop('num_train_samples', [])
        assert len(num_train_samples) == len(participants), 'Please provide the keyword argument num_train_samples, ' \
                                                            'containing the number of training samples for each ' \
                                                            'participant'
        num_total_samples = sum(num_train_samples)

        aggregated_model_state = None
        for num_samples, participant in zip(num_train_samples, participants):
            aggregated_model_state = add_weighted_model(aggregated_model_state,
                                                        load_participant_model_state(participant),
                                                        num_samples, num_total_samples)

        self._model.load_state_dict(aggregated_model_state)
        self.total_train_sample_num = num_total_samples
        self.save_model_state()


class CNNLightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, optimizer_args: OptimizerArgs, only_digits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNN_DropOut(only_digits=only_digits)
        self.optimizer_args = optimizer_args
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        o = self.optimizer_args
        return o.optimizer_class(self.model.parameters(), *o.optimizer_args, **o.optimizer_kwargs)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # TODO: this should actually be calculated on a validation set (missing cross entropy implementation)
        self.log('train/acc/{}'.format(self.participant_name), self.accuracy(preds, y).item())
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f'test/acc/{self.participant_name}', self.accuracy(preds, y).item())
        self.log(f'test/loss/{self.participant_name}', loss.item())
        return loss
