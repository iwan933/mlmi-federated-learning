import argparse
from typing import Dict, Optional

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import Tensor, optim

from mlmi.clustering import RandomClusterPartitioner
from mlmi.struct import ExperimentContext
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
    experiment_logger.experiment.add_scalar('test/loss/{}/mean'.format(model_name), torch.mean(loss), global_step=global_step)
    experiment_logger.experiment.add_histogram('test/acc/{}'.format(model_name), acc, global_step=global_step)
    experiment_logger.experiment.add_scalar('test/acc/{}/mean'.format(model_name), torch.mean(acc), global_step=global_step)


def initialize_clients(context: ExperimentContext, initial_model_state: Dict[str, Tensor]):
    clients = []
    for c, dataset in context.dataset.train_data_local_dict.items():
        client = FedAvgClient(str(c), context.model_args, context, context.dataset.train_data_local_dict[c],
                              context.dataset.data_local_train_num_dict[c], context.dataset.test_data_local_dict[c],
                              context.dataset.data_local_test_num_dict[c], context.experiment_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        client.overwrite_model_state(initial_model_state)
        clients.append(client)
    return clients


def run_fedavg(context: ExperimentContext, num_rounds: int, save_states: bool,
               initial_model_state: Optional[Dict[str, Tensor]] = None):
    server = FedAvgServer('initial_server', context.model_args, context)
    if initial_model_state is not None:
        server.overwrite_model_state(initial_model_state)
    clients = initialize_clients(context, server.model.state_dict())
    num_train_samples = [client.num_train_samples for client in clients]
    context.experiment_logger.experiment.add_histogram('sample/distribution', num_train_samples, global_step=0)

    for i in range(num_rounds):
        logger.info('starting training round {0}'.format(str(i + 1)))
        # train
        run_fedavg_round(server, clients, context.train_args, num_train_samples=num_train_samples)
        # test
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        # log and save
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), context.experiment_logger, i)
        if save_states:
            save_fedavg_state(context, i, server.model.state_dict())
        logger.info('finished training round')
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


def create_femnist_experiment_context(name: str, local_epochs: int, num_clients: int, batch_size: int, lr: float,
                                      client_fraction: float):
    optimizer_args = OptimizerArgs(optim.SGD, lr=lr)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(max_steps=local_epochs, gpus=1)
    else:
        training_args = TrainArgs(max_steps=local_epochs)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=batch_size)
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

        if args.hierarchical:
            context = create_femnist_experiment_context(name='fedavg_hierarchical', client_fraction=1.0, local_epochs=1,
                                                        lr=0.3, batch_size=10, num_clients=3400)
            run_fedavg_hierarchical(context, 1, 20)
        else:
            context = create_femnist_experiment_context(name='fedavg_default', client_fraction=1.0, local_epochs=1,
                                                        lr=0.3, batch_size=10, num_clients=3400)
            run_fedavg(context, 1, save_states=True)

    run()
