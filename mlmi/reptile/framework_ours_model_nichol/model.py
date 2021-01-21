from typing import Dict, List
from collections import OrderedDict
import copy

import torch
from torch import Tensor
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

import tensorflow.compat.v1 as tf

from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.struct import TrainArgs, ModelArgs, ExperimentContext, OptimizerArgs

import mlmi.reptile.framework_ours_model_nichol.variables as variables


logger = getLogger(__name__)

def weight_model(model: Dict[str, Tensor], num_samples: int, num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = OrderedDict()
    for key, w in model.items():
        weighted_model_state[key] = (num_samples / num_total_samples) * w
    return weighted_model_state

def sum_model_states(model_state_list: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
    result_state = copy.deepcopy(model_state_list[0])
    for model_state in model_state_list[1:]:
        for key, w in model_state.items():
            result_state[key] += w
    return result_state

def subtract_model_states(minuend: OrderedDict,
                          subtrahend: OrderedDict) -> OrderedDict:
    """
    Returns difference of two model_states: minuend - subtrahend
    """
    result_state = copy.deepcopy(minuend)
    for key, w in subtrahend.items():
        result_state[key] -= w
    return result_state

class ReptileClient(BaseTrainingParticipant):

    def __init__(self, client_id: str, *args, **kwargs):
        self._name = client_id
        with tf.variable_scope(self._name):
            super().__init__(client_id=client_id, *args, **kwargs)

    def save_model_state(self):
        """
        Saves the model state of the aggregated model
        :param target_path: The path to save the model at
        :return:
        """
        # Do not save model state (not enough disc space for thousands of model states)
        # torch.save(self._model.state_dict(), self.get_checkpoint_path())

    def overwrite_model_state(self, model_state):
        pass

    def get_model_state(self):
        pass

    def train(self):
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
        new_model_state = self.model.state_dict()
        for key, w in new_model_state.items():
            if key.endswith('running_mean') or key.endswith('running_var') \
                or key.endswith('num_batches_tracked'):
                # Do not update non-trainable batch norm parameters
                continue
            new_model_state[key] = w + learning_rate * gradient[key]
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

    def forward(self, x):
        return self.model(x)


class OmniglotModel(torch.nn.Module):
    """
    A model for Omniglot classification. Adapted from Nichol 2018.
    """

    def __init__(self, num_classes: int):
        super().__init__()

        # The below layers could be more conveniently generated as an array.
        # However, the class function state_dict() does not work then. For this
        # reason, we define all layers individually.
        self.conv2d_1 = self._make_conv2d_layer(first=True)
        self.batchnorm_1 = self._make_batchnorm_layer()
        self.relu_1 = self._make_relu_layer()

        self.conv2d_2 = self._make_conv2d_layer()
        self.batchnorm_2 = self._make_batchnorm_layer()
        self.relu_2 = self._make_relu_layer()

        self.conv2d_3 = self._make_conv2d_layer()
        self.batchnorm_3 = self._make_batchnorm_layer()
        self.relu_3 = self._make_relu_layer()

        self.conv2d_4 = self._make_conv2d_layer()
        self.batchnorm_4 = self._make_batchnorm_layer()
        self.relu_4 = self._make_relu_layer()

        self.flatten = torch.nn.Flatten(start_dim=1)
        self.logits = torch.nn.Linear(in_features=256, out_features=num_classes)

    def _make_conv2d_layer(self, first: bool = False):
        kernel_size = 3
        return torch.nn.Conv2d(
            in_channels=1 if first else 64,
            out_channels=64,
            kernel_size=kernel_size,
            stride=2,
            padding=int((kernel_size - 1) / 2)  # Apply same padding
        )

    def _make_batchnorm_layer(self):
        return torch.nn.BatchNorm2d(
            num_features=64, eps=1e-3, momentum=0.01, track_running_stats=False
        )

    def _make_relu_layer(self):
        return torch.nn.ReLU()

    def forward(self, x):
        x = self.conv2d_1(x)
        x = self.batchnorm_1(x)
        x = self.relu_1(x)

        x = self.conv2d_2(x)
        x = self.batchnorm_2(x)
        x = self.relu_2(x)

        x = self.conv2d_3(x)
        x = self.batchnorm_3(x)
        x = self.relu_3(x)

        x = self.conv2d_4(x)
        x = self.batchnorm_4(x)
        x = self.relu_4(x)

        x = self.flatten(x)
        x = self.logits(x)
        return x
