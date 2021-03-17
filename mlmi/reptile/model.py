from typing import Dict, List, Optional
from collections import OrderedDict
import copy

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl

from mlmi.log import getLogger
from mlmi.participant import BaseTrainingParticipant, BaseAggregatorParticipant
from mlmi.reptile.structs import ReptileExperimentContext


logger = getLogger(__name__)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)


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


def subtract_model_states(minuend: Dict[str, Tensor],
                          subtrahend: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """
    Returns difference of two model_states: minuend - subtrahend
    """
    result_state = copy.deepcopy(minuend)
    for key, w in subtrahend.items():
        result_state[key] -= w
    return result_state


def estimate_weights(labels):
    weight = torch.ones((7,))
    label, counts = torch.unique(labels, return_counts=True)
    weight[label] = weight[label] - counts / torch.sum(counts)
    return weight


class ReptileClient(BaseTrainingParticipant):

    def __init__(self, do_balancing, **kwargs):
        self.do_balancing = do_balancing
        super().__init__(**kwargs)

    def get_model_kwargs(self) -> Optional[Dict]:
        if not self.do_balancing:
            return None
        labels = torch.LongTensor([])
        for _, y in self.train_data_loader:
            labels = torch.cat((labels, y))
        for _, y in self.test_data_loader:
            labels = torch.cat((labels, y))
        weights = estimate_weights(labels)
        return {
            'weights': weights
        }

    def create_trainer(self, enable_logging=True, **kwargs) -> pl.Trainer:
        """
        Creates a new trainer instance for each training round.
        :param kwargs: additional keyword arguments to send to the trainer for configuration
        :return: a pytorch lightning trainer instance
        """
        _kwargs = kwargs.copy()
        _kwargs['logger'] = self.logger
        if torch.cuda.is_available():
            _kwargs['gpus'] = 1
        return pl.Trainer(
            checkpoint_callback=False,
            limit_val_batches=0.0,
            **_kwargs
        )

    def overwrite_model_state(self, model_state: Dict[str, Tensor]):
        """
        Loads the model state into the current model instance
        :param model_state: The model state to load
        """
        self.model.load_state_dict(copy.deepcopy(model_state), strict=False)

    def save_model_state(self):
        """
        Saves the model state of the aggregated model
        :param target_path: The path to save the model at
        :return:
        """
        # Do not save model state (not enough disc space for thousands of model states)
        # torch.save(self._model.state_dict(), self.get_checkpoint_path())


class ReptileServer(BaseAggregatorParticipant):
    def __init__(self,
                 participant_name: str,
                 model_args,
                 context: ReptileExperimentContext,
                 initial_model_state: OrderedDict = None):
        super().__init__(participant_name, model_args, context)
        # Initialize model parameters
        if initial_model_state is not None:
            self.model.load_state_dict(initial_model_state)

    @property
    def model_args(self):
        return self._model_args

    def aggregate(self,
                  participants: List[ReptileClient],
                  meta_learning_rate: float,
                  weighted: bool = True):
        # Average participants' model states for meta_gradient of server
        initial_model_state = copy.deepcopy(self.model.cpu().state_dict())
        if weighted:
            # meta_gradient = weighted (by number of samples) average of
            # participants' model updates
            num_train_samples_total = sum([p._num_train_samples for p in participants])
            new_states = [
                weight_model(
                    model=p.model.cpu().state_dict(),
                    num_samples=p._num_train_samples,
                    num_total_samples=num_train_samples_total
                ) for p in participants
            ]
        else:
            # meta_gradient = simple average of participants' model updates
            new_states = [
                weight_model(
                    model=p.model.cpu().state_dict(),
                    num_samples=1,
                    num_total_samples=len(participants)
                ) for p in participants
            ]
        meta_gradient = subtract_model_states(
            minuend=sum_model_states(new_states),
            subtrahend=initial_model_state
        )

        # Update model state with meta_gradient using simple gradient descent
        self.update_model_state(meta_gradient, meta_learning_rate)

    def update_model_state(self, gradient, learning_rate):
        """
        Update model state with vanilla gradient descent
        :param gradient: OrderedDict[str, Tensor]
        :return:
        """
        new_model_state = copy.deepcopy(self.model.state_dict())
        for key, w in new_model_state.items():
            if key.endswith('running_mean') or key.endswith('running_var') \
                or key.endswith('num_batches_tracked'):
                # Do not update non-trainable batch norm parameters
                continue
            new_model_state[key] = w + learning_rate * gradient[key]
        self.model.load_state_dict(new_model_state)


def apply_same_padding(x, kernel_size, stride):
    if x.shape[2] % stride == 0:  # input image width % stride
        pad = max(kernel_size - stride, 0)
    else:
        pad = max(kernel_size - (x.shape[2] % stride), 0)

    if pad % 2 == 0:
        pad_val = pad // 2
        padding = (pad_val, pad_val, pad_val, pad_val)
    else:
        pad_val_start = pad // 2
        pad_val_end = pad - pad_val_start
        padding = (pad_val_start, pad_val_end, pad_val_start, pad_val_end)

    return F.pad(x, padding, "constant", 0)
