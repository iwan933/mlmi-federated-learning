from typing import Dict, List

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

def weight_model(model: Dict[str, Tensor], num_samples: int, num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = dict()
    for key, w in model.items():
        weighted_model_state[key] = (num_samples / num_total_samples) * w
    return weighted_model_state

class ReptileClient(BaseTrainingParticipant):
    def __init__(self):
        super().__init__()

    def train(self):
        # TODO: Sort out training routine. Include hyperparameters:
        #       * inner_learning_rate
        #       * inner_batch_size // is already taken care of by dataloader
        #       * inner_num_epochs
        #       * inner_optimizer
        trainer = self.create_trainer()

        # TODO: Do number of gradient steps (epochs) on local dataset


class ReptileServer(BaseAggregatorParticipant):
    def __init__(self, initial_model_params, num_meta_iterations, meta_learning_rate):
        super().__init__()

    def aggregate(self,
                  participants: List[BaseParticipant],
                  weighted: bool = True,
                  *args, **kwargs):

        # TODO: 1. Compute meta gradient
        meta_gradient = None
        if weighted:
            # TODO: meta_gradient = weighted average of participants' model updates
            num_train_samples: List[int] = kwargs.pop('num_train_samples', [])
            assert len(num_train_samples) == len(participants), 'Please provide the keyword argument num_train_samples, ' \
                                                                'containing the number of training samples for each ' \
                                                                'participant'
            weighted_model_state_list = []
            num_total_samples = sum(num_train_samples)
            for num_samples, participant in zip(num_train_samples, participants):
                weighted_model_state = weight_model(
                    participant.model.state_dict(), num_samples, num_total_samples
                )
            weighted_model_state_list.append(weighted_model_state)
            weighted_model_sum = sum_model_states(weighted_model_state_list)
            self._model.load_state_dict(weighted_model_sum)
            self.total_train_sample_num = num_total_samples
            self.save_model_state()
        else:
            # TODO: meta_gradient = simple average of participants' model updates
            pass

        # TODO: Update model state with meta_gradient using optimizer of choice