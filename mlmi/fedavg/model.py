from typing import Dict, List

from torch import Tensor, nn
from torch.nn import functional as F

import pytorch_lightning as pl

from fedml_api.model.cv.cnn import CNN_DropOut

from mlmi.log import getLogger
from mlmi.participant import BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.struct import OptimizerArgs


logger = getLogger(__name__)


def weigth_model(model: Dict[str, Tensor], num_samples: int, num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = dict()
    for key, w in model.items():
        weighted_model_state[key] = (num_samples / num_total_samples) * w
    return weighted_model_state


def sum_model_states(model_state_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    result_state = model_state_list[0]
    for model_state in model_state_list[1:]:
        for key, w in model_state.items():
            result_state[key] += w
    return result_state


def load_participant_model_state(participant: BaseParticipant) -> Dict[str, Tensor]:
    """
    Method to load the current model state from a participant
    :param participant: participant to load the model from
    :return:
    """
    return participant.get_model().state_dict()


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
        weighted_model_state_list = []
        num_total_samples = sum(num_train_samples)
        for num_samples, participant in zip(num_train_samples, participants):
            weighted_model_state = weigth_model(load_participant_model_state(participant),
                                                num_samples, num_total_samples)
            weighted_model_state_list.append(weighted_model_state)
        weighted_model_sum = sum_model_states(weighted_model_state_list)
        self.model.load_state_dict(weighted_model_sum)
        self.total_train_sample_num = num_total_samples


class CNNLightning(pl.LightningModule):

    def __init__(self, optimizer_args: OptimizerArgs, only_digits=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = CNN_DropOut(only_digits=only_digits)
        self.optimizer_args = optimizer_args

    def configure_optimizers(self):
        o = self.optimizer_args
        return o.optimizer_class(self.model.parameters(), *o.optimizer_args, **o.optimizer_kwargs)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        self.log('train/loss', loss)
        return loss
