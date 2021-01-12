from typing import Dict, List
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.struct import TrainArgs, ModelArgs, ExperimentContext, OptimizerArgs


logger = getLogger(__name__)

def weight_model(model: Dict[str, Tensor], num_samples: int, num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = OrderedDict()
    for key, w in model.items():
        weighted_model_state[key] = (num_samples / num_total_samples) * w
    return weighted_model_state

def sum_model_states(model_state_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    result_state = model_state_list[0].copy()
    for model_state in model_state_list[1:]:
        for key, w in model_state.items():
            result_state[key] += w
    return result_state

def subtract_model_states(minuend: OrderedDict,
                          subtrahend: OrderedDict) -> OrderedDict:
    """
    Returns difference of two model_states: minuend - subtrahend
    """
    result_state = minuend.copy()
    for key, w in subtrahend.items():
        result_state[key] -= w
    return result_state

class ReptileClient(BaseTrainingParticipant):
    pass


class ReptileServer(BaseAggregatorParticipant):
    def __init__(self,
                 participant_name: str,
                 model_args,
                 context: ExperimentContext,
                 initial_model_state: OrderedDict = None):
        super().__init__(participant_name, model_args, context)
        # Initialize model parameters
        if initial_model_state is not None:
            self.model.load_state_dict(initial_model_state)

    @property
    def model_args(self):
        return self._model_args

    def aggregate(self,
                  participants: List[BaseParticipant],
                  meta_learning_rate: float,
                  weighted: bool = True):

        # Collect participants' model states and calculate model differences to
        # initial model (= model deltas)
        initial_model_state = self.model.state_dict()
        participant_model_deltas = []
        for participant in participants:
            participant_model_deltas.append(
                subtract_model_states(
                    participant.model.state_dict(), initial_model_state
                )
            )
        if weighted:
            # meta_gradient = weighted (by number of samples) average of
            # participants' model updates
            num_train_samples = []
            for participant in participants:
                num_train_samples.append(participant.num_train_samples)
            weighted_model_delta_list = []
            num_total_samples = sum(num_train_samples)
            for num_samples, pmd in zip(num_train_samples, participant_model_deltas):
                weighted_model_delta = weight_model(
                    pmd, num_samples, num_total_samples
                )
                weighted_model_delta_list.append(weighted_model_delta)
            meta_gradient = sum_model_states(weighted_model_delta_list)
            self.total_train_sample_num = num_total_samples
        else:
            # meta_gradient = simple average of participants' model updates
            scaled_model_delta_list = []
            for pmd in participant_model_deltas:
                scaled_model_delta = weight_model(
                    pmd, 1, len(participant_model_deltas)
                )
                scaled_model_delta_list.append(scaled_model_delta)
            meta_gradient = sum_model_states(scaled_model_delta_list)

        # Update model state with meta_gradient using simple gradient descent
        self.update_model_state(meta_gradient, meta_learning_rate)

    def update_model_state(self, gradient, learning_rate):
        """
        Update model state with vanilla gradient descent
        :param gradient: OrderedDict[str, Tensor]
        :return:
        """
        # TODO (optional): Extend this function with other optimizer options
        #                  than vanilla GD
        new_model_state = self.model.state_dict().copy()
        for key, w in new_model_state.items():
            new_model_state[key] = w + \
                learning_rate * gradient[key]
        self.model.load_state_dict(new_model_state)


class OmniglotLightning(BaseParticipantModel, pl.LightningModule):
    """
    A model for Omniglot classification - PyTorch implementation.
    """
    def __init__(self, optimizer_args: OptimizerArgs, num_classes:int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = OmniglotModel(num_classes=num_classes)
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
        self.log('train/acc/{}'.format(self.participant_name), self.accuracy(preds, y))
        self.log('train/loss/{}'.format(self.participant_name), loss)
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('test/acc/{}'.format(self.participant_name), self.accuracy(preds, y))
        self.log('test/loss/{}'.format(self.participant_name), loss)
        return loss


class OmniglotModel(torch.nn.Module):
    """
    A model for Omniglot classification. Adapted from Nichol 2018.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.conv2d = []
        self.batchnorm = []
        self.relu = []

        kernel_size = 3
        for i in range(4):
            self.conv2d.append(
                torch.nn.Conv2d(
                    in_channels=1 if i == 0 else 64,
                    out_channels=64,
                    kernel_size=kernel_size,
                    stride=2,
                    padding=int((kernel_size - 1) / 2)  # Apply same padding
                )
            )
            self.batchnorm.append(
                torch.nn.BatchNorm2d(
                    num_features=64,
                    eps=1e-3,
                    momentum=0.01
                )
            )
            self.relu.append(
                torch.nn.ReLU()
            )
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.logits = torch.nn.Linear(in_features=256, out_features=num_classes)

    def forward(self, x):
        for i in range(4):
            x = self.conv2d[i](x)
            x = self.batchnorm[i](x)
            x = self.relu[i](x)
        x = self.flatten(x)
        x = self.logits(x)
        return x
