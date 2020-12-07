from typing import Dict, List

import torch
from torch import Tensor, nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from mlmi.log import getLogger
from mlmi.client import BaseClient
from mlmi.model.aggregator import BaseAggregator
from mlmi.model.trainer import BaseTrainer
from mlmi.server import BaseServer
from mlmi.struct import TrainArgs, TrainResults, ModelArgs, CriterionArgs


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


class FedAvgAggregator(BaseAggregator):
    def aggregate_models(self, model_to_update: Dict[str, Tensor], models_to_aggregate: List[Dict[str, Tensor]],
                         *args, **kwargs) -> Dict[str, Tensor]:
        num_train_samples: List[int] = kwargs.pop('num_train_samples', None)
        if num_train_samples is None:
            raise ValueError('No information on number of training samples per model passed to FedAvg aggregator.')

        assert len(num_train_samples) == len(models_to_aggregate), 'Please provide the num_train_samples for each model'

        num_total_samples: int = sum(num_train_samples)
        weigthed_model_state_list = []
        for num_samples, model_to_aggregate in zip(num_train_samples, models_to_aggregate):
            weigthed_model_state = weigth_model(model_to_aggregate, num_samples, num_total_samples)
            weigthed_model_state_list.append(weigthed_model_state)

        weigthed_model_sum = sum_model_states(weigthed_model_state_list)
        return weigthed_model_sum


class FedAvgTrainer(BaseTrainer):

    def train_model(self, model: nn.Module, dataloader: data.DataLoader, num_train_samples: int, training_args: TrainArgs,
                    optimizer: optim.Optimizer, criterion: nn.Module, tensorboard_writer: SummaryWriter) -> TrainResults:
        assert model.training, 'Model is not in training mode please set model to ' \
                               'training mode before running the trainer'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model.to(device)
        criterion.to(device)
        train_results = TrainResults(num_train=num_train_samples)
        iteration_losses = []
        epoch_losses = []
        for epoch in range(training_args.epochs):
            logger.debug('running epoch {0}'.format(str(epoch + 1)))
            iterations = 0
            epoch_loss = 0.0
            for x, y in dataloader:
                x, y = x.to(device), y.type(torch.LongTensor).to(device)
                model.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                iter_loss = loss.item()
                iteration_losses.append(iter_loss)
                epoch_loss += iter_loss
                iterations += 1
            epoch_losses.append(epoch_loss / iterations)
            logger.debug('finished epoch {0}'.format(str(epoch + 1)))
            train_results.add_results(epoch_loss)

        # write data to tensorboard
        for i, iter_loss in enumerate(iteration_losses):
            tensorboard_writer.add_scalar('Loss/iteration/train', iter_loss, i)
        for i, epoch_loss in enumerate(epoch_losses):
            tensorboard_writer.add_scalar('Loss/epoch/train', epoch_loss, i)

        train_results.set_model_state(model.cpu().state_dict())
        return train_results


class FedAvgClient(BaseClient):

    def __init__(self, client_id: str, train_dataloader: data.DataLoader, num_train_samples: int,
                 test_dataloader: data.DataLoader, num_test_samples: int, model_args: ModelArgs,
                 criterion_args: CriterionArgs, *args, **kwargs):
        super().__init__(client_id, train_dataloader, num_train_samples, test_dataloader, num_test_samples, model_args,
                         criterion_args, *args, **kwargs)

    def get_optimizer(self) -> optim.Optimizer:
        """
        Return None to pick the optimizer from the training arguments. Optionally a custom optimizer can be instantiated
        and persisted. This implementation should be used for optimizers like Adam, that adapt with each training round
        :return:
        """
        return None

    def get_tensorboard_writer(self) -> SummaryWriter:
        return self.tensorboard_writer

    def get_trainer(self) -> BaseTrainer:
        return FedAvgTrainer()


class FedAvgServer(BaseServer):

    def __init__(self, model_args: ModelArgs, train_args: TrainArgs):
        self.aggregator = FedAvgAggregator()
        self.model = model_args.model_class(*model_args.args, **model_args.kwargs)
        self.train_args = train_args

    def get_aggregator(self):
        return self.aggregator

    def aggregate(self, train_results: List[TrainResults], *args, **kwargs):
        models_to_aggregate = []
        num_train_samples = []
        for result in train_results:
            # append the final model resulting from training
            models_to_aggregate.append(result.model_state)
            # append number of training samples of the client
            num_train_samples.append(result.num_train)
        aggregated_model_state = self.aggregator.aggregate_models(None, models_to_aggregate,
                                                                  num_train_samples=num_train_samples)
        self.model.load_state_dict(aggregated_model_state)

    def get_model_state(self) -> Dict[str, Tensor]:
        return self.model.state_dict()

    def get_training_arguments(self):
        return self.train_args
