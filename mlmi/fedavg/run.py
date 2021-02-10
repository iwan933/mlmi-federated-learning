import argparse
import math
import numpy as np
from typing import Callable, Dict, List, Optional

from sklearn.model_selection import ParameterGrid

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import IntTensor, Tensor, optim
from torch.utils.data import DataLoader

from mlmi.fedavg.data import augment_for_clustering, scratch_data, non_iid_scratch
from mlmi.plot import generate_data_label_heatmap
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.participant import BaseTrainingParticipant
from mlmi.structs import ClusterArgs, FederatedDatasetData
from mlmi.clustering import ModelFlattenWeightsPartitioner
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset, load_mnist_dataset
from mlmi.fedavg.model import CNNMnistLightning, FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, \
    save_fedavg_state
from mlmi.structs import ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_global_model, fix_random_seeds

logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=False)
    parser.add_argument('--search-grid', dest='search_grid', action='store_const',
                        const=True, default=False)
    parser.add_argument('--cifar10', dest='cifar10', action='store_const',
                        const=True, default=False)
    parser.add_argument('--cifar100', dest='cifar100', action='store_const',
                        const=True, default=False)
    parser.add_argument('--log-data-distribution', dest='log_data_distribution', action='store_const',
                        const=True, default=False)
    parser.add_argument('--no-model-reuse', dest='load_last_state', action='store_const',
                        const=False, default=False)
    parser.add_argument('--scratch-data', dest='scratch_data', action='store_const',
                        const=True, default=False)
    parser.add_argument('--max-last', type=int, dest='max_last', default=-1)
    parser.add_argument('--gradient-clip-val', type=float, dest='gradient_clip_val', default=0.0)
    parser.add_argument('--briggs', dest='briggs', action='store_const',
                        const=True, default=False)
    parser.add_argument('--mnist', dest='mnist', action='store_const',
                        const=True, default=False)
    parser.add_argument('--show-progress-bar', dest='no_progress_bar', action='store_const',
                        const=False, default=True)
    parser.add_argument('--seed', dest='seed', type=int, default=123123123)
    parser.add_argument('--plot-client-labels', dest='plot_client_labels', action='store_const', default=False,
                        const=True)
    parser.add_argument('--non-iid-scratch', dest='non_iid_scratch', action='store_const', default=False,
                        const=True)


def initialize_clients(context: 'FedAvgExperimentContext', dataset: 'FederatedDatasetData',
                       initial_model_state: Dict[str, Tensor]):
    clients = []
    logger.debug('... creating total of {} clients'.format(len(dataset.train_data_local_dict.items())))
    for i, (c, _) in enumerate(dataset.train_data_local_dict.items()):
        client = FedAvgClient(str(c), context.model_args, context, dataset.train_data_local_dict[c],
                              dataset.data_local_train_num_dict[c], dataset.test_data_local_dict[c],
                              dataset.data_local_test_num_dict[c], context.experiment_logger)
        client.overwrite_model_state(initial_model_state)
        clients.append(client)
        if (i + 1) % 50 == 0:
            logger.debug('... created {}/{}'.format(i + 1, len(dataset.train_data_local_dict.items())))
    return clients


def log_data_distribution_by_dataset(name: str, dataset: FederatedDatasetData, experiment_logger: LightningLoggerBase,
                                     global_step=0):
    dataloaders = [d for d in dataset.train_data_local_dict.values()]
    log_data_distribution_by_dataloaders(name, dataset.class_num, dataloaders, experiment_logger, global_step)


def log_data_distribution_by_participants(name: str, num_classes: int,
                                          training_participants: List['BaseTrainingParticipant'],
                                          experiment_logger: LightningLoggerBase, global_step=0):
    dataloaders = [p.train_data_loader for p in training_participants]
    log_data_distribution_by_dataloaders(name, num_classes, dataloaders, experiment_logger, global_step)


def log_data_distribution_by_dataloaders(name: str, num_classes: int, dataloaders: List[DataLoader],
                                         experiment_logger: LightningLoggerBase, global_step=0):
    num_partitions = len(dataloaders)
    num_train_samples = np.array([], dtype=np.int)
    label_distribution = np.zeros((num_classes,))

    logger.debug('... extracting dataset distributions')
    for i, dataloader in enumerate(dataloaders):
        local_labels = None
        num_all_samples = 0
        for x, y in dataloader:
            num_all_samples += x.shape[0]
            if local_labels is None:
                local_labels = y
            else:
                local_labels = torch.cat((local_labels, y), dim=0)
        unique_labels, counts = torch.unique(local_labels, return_counts=True)
        for u, c in zip(unique_labels, counts):
            label_distribution[u] += c
        num_train_samples = np.append(num_train_samples, num_all_samples)
        if (i + 1) % 50 == 0:
            logger.debug('... extracted {}/{} partitions'.format(i + 1, num_partitions))

    unique_num_samples, counts = torch.unique(IntTensor(num_train_samples), return_counts=True)
    num_train_samples = np.zeros((torch.max(unique_num_samples) + 1,))
    for u, c in zip(unique_num_samples, counts):
        num_train_samples[u] = c
    chunks = int(len(num_train_samples) / 10)
    num_10step_split = [np.sum(num_train_samples[s * 10:(s + 1) * 10]) for s in range(chunks)]
    if int(len(num_train_samples) / 10) < len(num_train_samples) / 10:
        num_10step_split.append(np.sum(num_train_samples[chunks * 10:]))

    for i, v in enumerate(label_distribution):
        experiment_logger.experiment.add_scalar(f'distribution/label/{name}', v, global_step=i)

    for i, v in enumerate(num_10step_split):
        experiment_logger.experiment.add_scalar(f'distribution/sample/{name}', v, global_step=i)


def run_fedavg(
        context: FedAvgExperimentContext,
        num_rounds: int,
        save_states: bool,
        dataset: 'FederatedDatasetData',
        initial_model_state: Optional[Dict[str, Tensor]] = None,
        clients: Optional[List['FedAvgClient']] = None,
        server: Optional['FedAvgServer'] = None,
        start_round=0,
        restore_state=False,
        after_round_evaluation: Optional[List[Callable]] = None
):
    assert (server is None and clients is None) or (server is not None and clients is not None)

    if clients is None or server is None:
        logger.info('initializing server ...')
        server = FedAvgServer('initial_server', context.model_args, context)
        if initial_model_state is not None:
            server.overwrite_model_state(initial_model_state)
        logger.info('initializing clients ...')
        clients = initialize_clients(context, dataset, server.model.state_dict())

    if start_round + 1 > num_rounds:
        return server, clients

    num_train_samples = [client.num_train_samples for client in clients]
    num_total_samples = sum(num_train_samples)
    logger.info(f'... copied {num_total_samples} data samples in total')

    for i in range(start_round, num_rounds):
        logger.info(f'starting training round {i + 1} ...')
        round_model_state = load_fedavg_state(context, i + 1)
        if restore_state and round_model_state is not None:
            logger.info(f'skipping training and loading model from disk ...')
            server.overwrite_model_state(round_model_state)
        else:
            run_fedavg_round(server, clients, context.train_args, client_fraction=context.client_fraction)
        # test over all clients
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        loss, acc = result.get('test/loss'), result.get('test/acc')
        if after_round_evaluation is not None:
            for c in after_round_evaluation:
                c(loss, acc, i)
        logger.info(
            f'... finished training round (mean loss: {torch.mean(loss):.2f}, mean acc: {torch.mean(acc):.2f})')
        # log and save
        if save_states:
            save_fedavg_state(context, i + 1, server.model.state_dict())
    return server, clients


def create_femnist_experiment_context(name: str, local_epochs: int, batch_size: int, lr: float, client_fraction: float,
                                      dataset_name: str, fixed_logger_version=None, no_progress_bar=False,
                                      cluster_args: Optional[ClusterArgs] = None):
    logger.debug('creating experiment context ...')
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(CNNLightning, optimizer_args=optimizer_args, only_digits=False)
    train_args_dict = {
        'max_epochs': local_epochs,
        'min_epochs': local_epochs
    }
    if no_progress_bar:
        train_args_dict['progress_bar_refresh_rate'] = 0
    training_args = TrainArgs(**train_args_dict)
    context = FedAvgExperimentContext(name=name, client_fraction=client_fraction, local_epochs=local_epochs,
                                      lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                      model_args=model_args, train_args=training_args, dataset_name=dataset_name)
    experiment_specification = f'{context}'
    if cluster_args is not None:
        context.cluster_args = cluster_args
        experiment_specification += f'_{cluster_args}'
    experiment_logger = create_tensorboard_logger(context.name, experiment_specification, fixed_logger_version)
    context.experiment_logger = experiment_logger
    return context


def create_mnist_experiment_context(name: str, local_epochs: int, batch_size: int, lr: float, client_fraction: float,
                                    dataset_name: str, num_classes: int, fixed_logger_version=None,
                                    no_progress_bar=False, cluster_args: Optional[ClusterArgs] = None):
    logger.debug('creating experiment context ...')
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(CNNMnistLightning, num_classes=num_classes, optimizer_args=optimizer_args)
    train_args_dict = {
        'max_epochs': local_epochs,
        'min_epochs': local_epochs
    }
    if no_progress_bar:
        train_args_dict['progress_bar_refresh_rate'] = 0
    training_args = TrainArgs(**train_args_dict)
    context = FedAvgExperimentContext(name=name, client_fraction=client_fraction, local_epochs=local_epochs,
                                      lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                      model_args=model_args, train_args=training_args, dataset_name=dataset_name)
    experiment_specification = f'{context}'
    experiment_specification += f'_{optimizer_args}'
    if cluster_args is not None:
        context.cluster_args = cluster_args
        experiment_specification += f'_{cluster_args}'
    experiment_logger = create_tensorboard_logger(context.name, experiment_specification, fixed_logger_version)
    context.experiment_logger = experiment_logger
    return context


def load_last_state_for_configuration(context: FedAvgExperimentContext, max_last: int = -1):
    last_round = -1
    last_state = None
    # check if saved model for given experiment and round already exists
    while True:
        saved_state = load_fedavg_state(context, last_round + 1)
        if saved_state is not None:
            last_state = saved_state
            last_round += 1
            if not max_last == -1 and max_last <= last_round:
                return last_state, last_round
            logger.info(f'found saved state for {context}, round {last_round}')
        else:
            return last_state, last_round


def lr_gen(bases: List[float], powers: List[int]):
    for x in bases:
        for y in powers:
            yield x * math.pow(10, y)


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        # fix_random_seeds(args.seed)

        logger.debug('loading experiment data ...')
        data_dir = REPO_ROOT / 'data'
        fed_dataset = None
        context = None

        if args.cifar10:
            pass
        elif args.cifar100:
            pass
        elif args.mnist:
            fed_dataset = load_mnist_dataset(str(data_dir.absolute()), num_clients=100, batch_size=10)
        else:
            # default to femnist dataset
            fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=367, batch_size=10,
                                               only_digits=False, sample_threshold=250)

        if args.non_iid_scratch:
            non_iid_scratch(fed_dataset, num_mnist_label_zero=5)

        if args.scratch_data:
            client_fraction_to_scratch = 0.75
            data_fraction_to_scratch = 0.9
            scratch_data(fed_dataset, client_fraction_to_scratch=client_fraction_to_scratch,
                         fraction_to_scratch=data_fraction_to_scratch)
            fed_dataset.name += f'_scratched{client_fraction_to_scratch:.2f}by{data_fraction_to_scratch:.2f}'

        if args.log_data_distribution:
            logger.info('... found log distribution flag, only logging data distribution information')
            experiment_logger = create_tensorboard_logger('datadistribution', fed_dataset.name, version=0)
            log_data_distribution_by_dataset('fedavg', fed_dataset, experiment_logger)
            return

        if args.plot_client_labels:
            augment_for_clustering(fed_dataset, 0.1, 4, label_core_num=12, label_deviation=3)
            image = generate_data_label_heatmap('initial distribution', fed_dataset.train_data_local_dict.values(), 62)
            experiment_logger = create_tensorboard_logger('datadistribution', fed_dataset.name)
            experiment_logger.experiment.add_image('label distribution/test', image.numpy())
            return

        """
        default: run fed avg with fixed parameters
        """
        try:
            context = create_mnist_experiment_context(name='fedavg', client_fraction=0.1,
                                                      local_epochs=5, num_classes=10,
                                                      lr=0.1, batch_size=fed_dataset.batch_size,
                                                      dataset_name='mnist_momentum0.5',
                                                      no_progress_bar=args.no_progress_bar)
            logger.info(f'running FedAvg with the following configuration: {context}')
            run_fedavg(context, 50, save_states=False, dataset=fed_dataset)
        except Exception as e:
            logger.exception(f'Failed to execute configuration {context}', e)


    run()
