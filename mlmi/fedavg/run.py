import argparse
import math
import random
from typing import Dict, List, Optional

import numpy as np
from sklearn.model_selection import ParameterGrid

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import Tensor, optim
from torch.distributions import Categorical
from torch.utils import data as data

from mlmi.fedavg.data import scratch_data, select_random_fed_dataset_partitions
from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.selectors import sample_randomly_by_fraction
from mlmi.structs import ClusterArgs, FederatedDatasetData
from mlmi.clustering import RandomClusterPartitioner, GradientClusterPartitioner
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import load_fedavg_hierarchical_cluster_configuration, \
    load_fedavg_hierarchical_cluster_model_state, load_fedavg_state, run_fedavg_round, \
    save_fedavg_hierarchical_cluster_configuration, save_fedavg_hierarchical_cluster_model_state, save_fedavg_state, \
    run_train_round
from mlmi.structs import ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_global_model, fix_random_seeds, evaluate_local_models, \
    overwrite_participants_models

logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=True)
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
    parser.add_argument('--briggs', dest='briggs', action='store_const',
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
    experiment_logger.experiment.add_histogram('test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/acc/{}/mean'.format(model_name), torch.mean(acc),
                                            global_step=global_step)
    if loss.dim() == 0:
        loss = torch.tensor([loss])
    for x in loss:
        if torch.isnan(x) or torch.isinf(x):
            return
    experiment_logger.experiment.add_histogram('test/loss/{}'.format(model_name), loss, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/loss/{}/mean'.format(model_name), torch.mean(loss),
                                            global_step=global_step)


def log_goal_test_acc(model_name: str, acc: torch.Tensor,
                      experiment_logger: LightningLoggerBase, global_step: int):
    if acc.dim() == 0:
        acc = torch.tensor([acc])
    over80 = acc[acc >= 0.80]
    percentage = over80.shape[0] / acc.shape[0]
    experiment_logger.experiment.add_scalar('test/80/{}'.format(model_name), percentage, global_step=global_step)


def initialize_clients(context: 'FedAvgExperimentContext', dataset: 'FederatedDatasetData', initial_model_state: Dict[str, Tensor]):
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


def log_data_distribution(dataset: FederatedDatasetData, experiment_logger: LightningLoggerBase):
    num_partitions = len(dataset.train_data_local_dict.items())
    num_train_samples = []
    num_different_labels = []
    entropy_per_partition = []

    logger.debug('... extracting dataset distributions')
    for i, (c, dataloader) in enumerate(dataset.train_data_local_dict.items()):
        local_labels = None
        num_all_samples = 0
        for x, y in dataloader:
            num_all_samples += x.shape[0]
            if local_labels is None:
                local_labels = y
            else:
                local_labels = torch.cat((local_labels, y), dim=0)
        unique_labels, counts = torch.unique(local_labels, return_counts=True)
        entropy = Categorical(probs=counts / counts.sum()).entropy()
        num_different_labels.append(len(unique_labels))
        num_train_samples.append(num_all_samples)
        entropy_per_partition.append(entropy)

        if (i + 1) % 50 == 0:
            logger.debug('... extracted {}/{} partitions'.format(i + 1, num_partitions))

    experiment_logger.experiment.add_histogram('distribution/sample_num', Tensor(num_train_samples),
                                               global_step=0)
    experiment_logger.experiment.add_histogram('distribution/labels_num', Tensor(num_different_labels),
                                               global_step=0)
    experiment_logger.experiment.add_histogram('distribution/entropy', Tensor(entropy_per_partition),
                                               global_step=0)


def run_fedavg(context: FedAvgExperimentContext, num_rounds: int, save_states: bool, dataset: 'FederatedDatasetData',
               initial_model_state: Optional[Dict[str, Tensor]] = None, clients: Optional[List['FedAvgClient']] = None,
               server: Optional['FedAvgServer'] = None, start_round=0):
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
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), context.experiment_logger, i)
        logger.info('... finished training round')
    return server, clients


def run_fedavg_hierarchical(context: FedAvgExperimentContext, num_rounds_init: int, num_rounds_cluster: int,
                            dataset: 'FederatedDatasetData', restore_clustering=False, restore_fedavg=False):
    assert context.cluster_args is not None, 'Please set cluster args to run hierarchical experiment'

    saved_model_state = load_fedavg_state(context, num_rounds_init - 1)

    if saved_model_state is None or not restore_fedavg:
        server, clients = run_fedavg(context, num_rounds_init, save_states=True, dataset=dataset)
    else:
        server = FedAvgServer('initial_server', context.model_args, context)
        server.overwrite_model_state(saved_model_state)
        clients = initialize_clients(context, dataset, initial_model_state=saved_model_state)

    logger.debug('starting local training before clustering.')
    overwrite_participants_models(server.model.state_dict(), clients)
    run_train_round(clients, context.train_args)


    cluster_ids, cluster_clients = None, None
    if restore_clustering:
        # load configuration
        cluster_ids, cluster_clients = load_fedavg_hierarchical_cluster_configuration(context)

    if cluster_ids is not None and cluster_clients is not None:
        cluster_clients_dic = dict()
        for cluster_id, _clients in cluster_clients.items():
            cluster_clients_dic[cluster_id] = [c for c in clients if c._name in _clients]
    else:
        # Clustering of participants by model updates
        partitioner = context.cluster_args.partitioner_class(*context.cluster_args.args, **context.cluster_args.kwargs)
        cluster_clients_dic = partitioner.cluster(clients)
        _cluster_clients_dic = dict()
        for cluster_id, participants in cluster_clients_dic.items():
            _cluster_clients_dic[cluster_id] = [c._name for c in participants]
        save_fedavg_hierarchical_cluster_configuration(context, list(cluster_clients_dic.keys()), _cluster_clients_dic)

    eval_result = evaluate_global_model(global_model_participant=server, participants=clients)
    acc = eval_result.get('test/acc')
    loss = eval_result.get('test/loss')
    log_loss_and_acc('post clustering', loss, acc, context.experiment_logger, num_rounds_init)
    log_goal_test_acc('post clustering', acc, context.experiment_logger, num_rounds_init)


    # Initialize cluster models
    cluster_server_dic = {}
    for cluster_id, participants in cluster_clients_dic.items():
        cluster_server = FedAvgServer('cluster_server' + cluster_id, context.model_args, context)
        cluster_server.overwrite_model_state(server.model.state_dict())
        cluster_server_dic[cluster_id] = cluster_server

    # Train in clusters
    for i in range(num_rounds_cluster):
        for cluster_id in cluster_clients_dic.keys():
            logger.info('starting training cluster {1} in round {0}'.format(str(i + 1), cluster_id))
            # train
            cluster_server = cluster_server_dic[cluster_id]
            cluster_clients = cluster_clients_dic[cluster_id]
            loaded_state = load_fedavg_hierarchical_cluster_model_state(context, num_rounds_init, cluster_id, i)
            if restore_clustering and loaded_state is not None:
                cluster_server.overwrite_model_state(loaded_state)
                logger.info(f'skipping training cluster {cluster_id} in round {i + 1}. loaded state from disk.')
            else:
                logger.info(f'starting training cluster {cluster_id} in round {i + 1}')
                num_train_samples = [client.num_train_samples for client in cluster_clients]
                run_fedavg_round(cluster_server, cluster_clients, context.train_args,
                                 num_train_samples=num_train_samples)
            # test
            result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
            log_loss_and_acc(f'cluster{cluster_id}', result.get('test/loss'), result.get('test/acc'),
                             context.experiment_logger, i)
            log_goal_test_acc(f'cluster{cluster_id}_80', result.get('test/acc'), context.experiment_logger, i)
            save_fedavg_hierarchical_cluster_model_state(context, num_rounds_init, cluster_id, i,
                                                         cluster_server.model.state_dict())
            logger.info(f'finished training cluster {cluster_id}')
        logger.info('testing clustering round results')
        global_losses = None
        global_acc = None
        for cluster_id, cluster_clients in cluster_clients_dic.items():
            cluster_server = cluster_server_dic[cluster_id]
            result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
            if global_losses is None:
                global_losses = result.get('test/loss')
                global_acc = result.get('test/acc')
            else:
                global_losses = torch.cat((global_losses, result.get('test/loss')), dim=0)
                global_acc = torch.cat((global_acc, result.get('test/acc')), dim=0)
        log_loss_and_acc('total hierarchical', global_losses, global_acc, context.experiment_logger,
                         num_rounds_init + i)
        log_goal_test_acc('total 80%', global_acc, context.experiment_logger, num_rounds_init + i)


def create_femnist_experiment_context(name: str, local_epochs: int, batch_size: int, lr: float, client_fraction: float,
                                      dataset_name: str, fixed_logger_version=None,
                                      cluster_args: Optional[ClusterArgs] = None):
    logger.debug('creating experiment context ...')
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    training_args = TrainArgs(max_epochs=local_epochs, min_epochs=local_epochs, gradient_clip_val=0.5)
    context = FedAvgExperimentContext(name=name, client_fraction=client_fraction, local_epochs=local_epochs,
                                      lr=lr, batch_size=batch_size, optimizer_args=optimizer_args,
                                      model_args=model_args, train_args=training_args, dataset_name=dataset_name)
    if cluster_args is not None:
        context.cluster_args = cluster_args
    experiment_logger = create_tensorboard_logger(context.name, str(context), fixed_logger_version)
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


def configuration_generator(total_rounds=50):
    for fraction in [0.1, 0.2, 0.5, 1.0]:
        for fedavg_rounds in [1, 3, 5, 10]:
            yield ({
                'client_fraction': fraction,
            }, {
                'num_rounds_init': fedavg_rounds,
                'num_rounds_cluster': total_rounds - fedavg_rounds
            })


if __name__ == '__main__':
    def run():
        # fix for experiment reproducability
        fix_random_seeds(123123123)

        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        logger.debug('loading experiment data ...')
        data_dir = REPO_ROOT / 'data'
        fed_dataset = None
        context = None

        if args.cifar10:
            pass
        elif args.cifar100:
            pass
        else:
            context = create_femnist_experiment_context(name='fedavg_hierarchical', client_fraction=0.2, local_epochs=3,
                                                        lr=0.1, batch_size=10, dataset_name='femnist')
            # default to femnist dataset
            fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=3400,
                                               batch_size=context.batch_size)
            # select 367 clients as in briggs paper
            fed_dataset = select_random_fed_dataset_partitions(fed_dataset, 367)

        if args.scratch_data:
            scratch_data(fed_dataset, client_fraction_to_scratch=0.75, fraction_to_scratch=0.9)

        if args.log_data_distribution:
            logger.info('... found log distribution flag, only logging data distribution information')
            experiment_logger = create_tensorboard_logger('datadistribution', fed_dataset.name, version=0)
            log_data_distribution(fed_dataset, experiment_logger)
            return

        assert context is not None, 'Please create a context before running experiment'

        if args.briggs:
            for (configuration, round_configuration) in configuration_generator(50):
                cluster_args = ClusterArgs(GradientClusterPartitioner, linkage_mech="ward", criterion="distance",
                                           dis_metric="euclidean", max_value_criterion=10.0, plot_dendrogram=False)

                context = create_femnist_experiment_context(name='fedavg_hierarchical', local_epochs=3, lr=0.1,
                                                            batch_size=10, **configuration,
                                                            dataset_name=fed_dataset.name, cluster_args=cluster_args)
                context.cluster_args = cluster_args
                run_fedavg_hierarchical(context, restore_clustering=False, restore_fedavg=True,
                                        dataset=fed_dataset, **round_configuration)
        elif args.hierarchical:
            cluster_args = ClusterArgs(GradientClusterPartitioner, linkage_mech="ward", criterion="distance",
                                       dis_metric="euclidean", max_value_criterion=10.0, plot_dendrogram=False)

            context = create_femnist_experiment_context(name='fedavg_hierarchical', client_fraction=0.2, local_epochs=3,
                                                        lr=0.1, batch_size=10, dataset_name=fed_dataset.name,
                                                        cluster_args=cluster_args)
            context.cluster_args = cluster_args
            run_fedavg_hierarchical(context, 10, 2, restore_clustering=False, restore_fedavg=True, dataset=fed_dataset)
        elif args.search_grid:
            param_grid = {'lr': list(lr_gen([1], [-1])) + list(lr_gen([1, 2.5, 5, 7.5], [-2])) +
                                list(lr_gen([5, 7.5], [-3])), 'local_epochs': [1, 5],
                          'client_fraction': [0.1, 0.5, 1.0]}
            for configuration in ParameterGrid(param_grid):
                try:
                    logger.info(f'running FedAvg with the following configuration: {configuration}')
                    context = create_femnist_experiment_context(name='fedavg_default', batch_size=10,
                                                                dataset_name=fed_dataset.name, fixed_logger_version=0,
                                                                **configuration)
                    if args.load_last_state:
                        last_state, last_round = load_last_state_for_configuration(context, args.max_last)
                    else:
                        last_state, last_round = None, -1
                    run_fedavg(context, 10, save_states=True, initial_model_state=last_state,
                               start_round=last_round + 1, dataset=fed_dataset)
                except Exception as e:
                    logger.exception(f'Failed to execute configuration {configuration}', e)
        else:
            """
            default: run fed avg with fixed parameters
            """
            try:
                logger.info(f'running FedAvg with the following configuration: {context}')
                if args.load_last_state:
                    last_state, last_round = load_last_state_for_configuration(context, args.max_last)
                else:
                    last_state, last_round = None, -1
                run_fedavg(context, 10, save_states=False, initial_model_state=last_state,
                           start_round=last_round + 1, dataset=fed_dataset)
            except Exception as e:
                logger.exception(f'Failed to execute configuration {context}', e)
    run()
