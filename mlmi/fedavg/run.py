import argparse
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import Tensor, optim

from mlmi.clustering import RandomClusterPartitioner
from mlmi.selectors import sample_randomly_by_fraction
from mlmi.struct import ExperimentContext, FederatedDatasetData
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import load_fedavg_state, run_fedavg_round, save_fedavg_state
from mlmi.struct import ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_global_model


logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=False)
    parser.add_argument('--search-grid', dest='search_grid', action='store_const',
                        const=True, default=False)


def log_loss_and_acc(model_name: str, loss: torch.Tensor, acc: torch.Tensor, experiment_logger: LightningLoggerBase,
                     global_step: int):
    """
    Logs the loss and accuracy in an histogram as well as scalar
    :param model_name: name for logging
    :param loss: loss tensor
    :param acc: acc tensor
    :param experiment_logger: lightning logger
    :param global_step: global step
    :return:
    """
    experiment_logger.experiment.add_histogram('test/loss/{}'.format(model_name), loss, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/loss/{}/mean'.format(model_name), torch.mean(loss),
                                            global_step=global_step)
    experiment_logger.experiment.add_histogram('test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/acc/{}/mean'.format(model_name), torch.mean(acc),
                                            global_step=global_step)


def initialize_clients(context: ExperimentContext, initial_model_state: Dict[str, Tensor]):
    clients = []
    logger.debug('... creating total of {} clients'.format(len(context.dataset.train_data_local_dict.items())))
    for i, (c, dataset) in enumerate(context.dataset.train_data_local_dict.items()):
        client = FedAvgClient(str(c), context.model_args, context, context.dataset.train_data_local_dict[c],
                              context.dataset.data_local_train_num_dict[c], context.dataset.test_data_local_dict[c],
                              context.dataset.data_local_test_num_dict[c], context.experiment_logger)
        client.overwrite_model_state(initial_model_state)
        clients.append(client)
        if (i + 1) % 50 == 0:
            logger.debug('... created {}/{}'.format(i + 1, len(context.dataset.train_data_local_dict.items())))
    return clients


def run_fedavg(context: ExperimentContext, num_rounds: int, save_states: bool,
               initial_model_state: Optional[Dict[str, Tensor]] = None, clients: Optional[List['FedAvgClient']] = None,
               server: Optional['FedAvgServer'] = None):
    assert (server is None and clients is None) or (server is not None and clients is not None)
    if clients is None or server is None:
        logger.info('initializing server ...')
        server = FedAvgServer('initial_server', context.model_args, context)
        if initial_model_state is not None:
            server.overwrite_model_state(initial_model_state)
        logger.info('initializing clients ...')
        clients = initialize_clients(context, server.model.state_dict())
    num_train_samples = [client.num_train_samples for client in clients]
    num_total_samples = sum(num_train_samples)
    logger.info(f'... copied {num_total_samples} data samples in total')
    context.experiment_logger.experiment.add_histogram('sample/distribution', Tensor(num_train_samples), global_step=0)

    for i in range(num_rounds):
        logger.info('sampling clients ...')
        round_participants = sample_randomly_by_fraction(clients, context.client_fraction)
        num_samples = [c.num_train_samples for c in round_participants]
        logger.info(f'... sampled {len(round_participants)} at fraction: {context.client_fraction:.2f}')
        logger.info(f'starting training round {i + 1} ...')
        # train and aggregate over fraction
        run_fedavg_round(server, round_participants, context.train_args, num_train_samples=num_samples)
        # test over all clients
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        # log and save
        if save_states:
            save_fedavg_state(context, i, server.model.state_dict())
        for x in result.get('test/loss'):
            if torch.isnan(x) or torch.isinf(x):
                raise Exception('Loss is Nan or Inf, aborting training.')
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), context.experiment_logger, i)
        logger.info('... finished training round')
    return server, clients


def run_fedavg_hierarchical(context: ExperimentContext, num_rounds_init: int, num_rounds_cluster: int):
    saved_model_state = load_fedavg_state(context, num_rounds_init)
    if saved_model_state is None:
        server, clients = run_fedavg(context, num_rounds_init, save_states=True)
    else:
        server = FedAvgServer('initial_server', context.model_args, context)
        server.overwrite_model_state(saved_model_state)
        clients = initialize_clients(context, initial_model_state=saved_model_state)

    # Clustering of participants by model updates
    partitioner = RandomClusterPartitioner()
    cluster_clients_dic = partitioner.cluster(clients)

    # Initialize cluster models
    cluster_server_dic = {}
    for cluster_id, participants in cluster_clients_dic.items():
        cluster_server = FedAvgServer('cluster_server' + cluster_id, context.model_args, context)
        cluster_server.overwrite_model_state(server.model.state_dict())
        cluster_server_dic[cluster_id] = cluster_server

    # Train in clusters
    for cluster_id in cluster_clients_dic.keys():
        for i in range(num_rounds_cluster):
            logger.info('starting training cluster {1} in round {0}'.format(str(i + 1), cluster_id))
            # train
            cluster_server = cluster_server_dic[cluster_id]
            cluster_clients = cluster_clients_dic[cluster_id]
            num_train_samples = [client.num_train_samples for client in cluster_clients]
            run_fedavg_round(cluster_server, cluster_clients, context.train_args, num_train_samples=num_train_samples)
            # test
            result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
            log_loss_and_acc('cluster{}'.format(cluster_id), result.get('test/loss'), result.get('test/acc'),
                             context.experiment_logger, i)

            logger.info('finished training cluster {0}'.format(cluster_id))


def create_femnist_experiment_context(name: str, local_epochs: int, fed_dataset: FederatedDatasetData, batch_size: int,
                                      lr: float, client_fraction: float):
    logger.debug('creating experiment context ...')
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    training_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs)
    context = ExperimentContext(name=name, client_fraction=client_fraction, local_epochs=local_epochs,
                                lr=lr, batch_size=batch_size, optimizer_args=optimizer_args, model_args=model_args,
                                train_args=training_args, dataset=fed_dataset)
    experiment_logger = create_tensorboard_logger(context)
    context.experiment_logger = experiment_logger
    return context


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        logger.debug('loading experiment data ...')
        data_dir = REPO_ROOT / 'data'
        fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=3400, batch_size=10)

        if args.hierarchical:
            context = create_femnist_experiment_context(name='fedavg_hierarchical', client_fraction=0.1, local_epochs=1,
                                                        lr=0.3, batch_size=10, fed_dataset=fed_dataset)
            run_fedavg_hierarchical(context, 1, 20)
        elif args.search_grid:
            param_grid = {'lr': np.logspace(-1.0, -3.0, num=10), 'local_epochs': [1, 5],
                          'client_fraction': [0.0, 0.1, 0.5, 1.0]}
            for configuration in ParameterGrid(param_grid):
                try:
                    logger.info(f'running FedAvg with the following configuration: {configuration}')
                    context = create_femnist_experiment_context(name='fedavg_default', batch_size=10,
                                                                fed_dataset=fed_dataset, **configuration)
                    if load_fedavg_state(context, 0) is not None:
                        logger.info(f'skipping configuration {configuration}')
                        continue
                    run_fedavg(context, 3, save_states=True)
                except Exception as e:
                    logger.exception(f'Failed to execute configuration {configuration}', e)
        else:
            logger.info('default implementation missing, please provide according arguments')
    run()
