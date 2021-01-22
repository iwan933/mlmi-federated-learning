import copy
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn, optim
from torch.utils import data

from mlmi.structs import OptimizerArgs


class BaseParticipantModel(nn.Module):

    def __init__(self, *args, participant_name=None, optimizer_args: Optional['OptimizerArgs'] = None, **kwargs):
        assert participant_name is not None, 'Please provide a participant name parameter in model args to identify' \
                                             'your model in logging'
        assert optimizer_args is not None, 'Optimizer args not set!'
        super().__init__()
        self.participant_name = participant_name
        self.optimizer_args = optimizer_args
        self._loop_log = []
        self._step_log = None

    def configure_optimizer(self) -> optim.Optimizer:
        """
        Optimizer needs to be instantiated after model has been moved to gpu
        :return: optimizer for model
        """
        return self.optimizer_args(self.parameters())

    def training_step(self, train_batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, test_batch, batch_idx):
        raise NotImplementedError()

    def log(self, key: str, value: any):
        if self._step_log is None:
            self._step_log = {key: value}
        else:
            self._step_log.update({key: value})

    def get_step_log(self):
        return self._step_log

    def flush_step_log(self):
        self._step_log = None


def fit(model: BaseParticipantModel, dataloader: data.DataLoader, max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None, optimizer_state: Optional[dict] = None, allow_gpu=True,
        gradient_clip_val=0.5, **kwargs) -> Tuple[List[dict], dict]:
    steps_generator = get_steps_iterator(max_epochs, max_steps)
    model.train()
    if allow_gpu and torch.cuda.is_available():
        model = model.cuda()
    optimizer = model.configure_optimizer()
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    accumulated_logs = []
    for batch_idx, (x, y) in steps_generator(dataloader):
        if allow_gpu and torch.cuda.is_available():
            x, y = x.cuda(), y.cuda()

        # execute step
        optimizer.zero_grad()

        loss = model.training_step((x, y), batch_idx)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_val, norm_type=2.0)

        optimizer.step()

        # collect logs
        accumulated_logs.append(model.get_step_log())
        model.flush_step_log()
    return accumulated_logs, optimizer_state_dict_to_cpu(optimizer.state_dict())


def test(model: BaseParticipantModel, dataloader: data.DataLoader,  allow_gpu=True,):
    steps_generator = get_steps_iterator(1, None)
    model.eval()
    if allow_gpu and torch.cuda.is_available():
        model = model.cuda()

    accumulated_logs = []
    with torch.no_grad():
        for batch_idx, (x, y) in steps_generator(dataloader):
            if allow_gpu and torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # execute step
            model.test_step((x, y), batch_idx)

            # collect logs
            accumulated_logs.append(model.get_step_log())
            model.flush_step_log()
    return accumulated_logs


def get_steps_iterator(max_epochs, max_steps) -> Callable[[data.DataLoader], List[Tuple[int, Tuple[Tensor, Tensor]]]]:
    assert max_epochs is not None or max_steps is not None, 'Please set max_epochs or max_steps'

    if max_epochs is None:
        max_epochs = max_steps

    def _get_steps(dataloader: data.DataLoader) -> List[Tuple[int, Tuple[Tensor, Tensor]]]:
        num_steps = 0
        for e in range(max_epochs):
            for batch_idx, (x, y) in enumerate(dataloader):
                yield batch_idx, (x, y)
                num_steps += 1
                if max_steps is not None and num_steps >= max_steps:
                    return

    return _get_steps


def optimizer_state_dict_to_cpu(optimizer_state_dict):
    c = copy.deepcopy(optimizer_state_dict)
    o = {}
    state_dict = c.get('state')
    r = {}
    for key, state in state_dict.items():
        s = {}
        for k, v in state.items():
            if torch.is_tensor(v):
                s[k] = v.cpu()
            else:
                s[k] = v
        r[key] = s
    o['state'] = r
    o['param_groups'] = c.get('param_groups')
    return o
