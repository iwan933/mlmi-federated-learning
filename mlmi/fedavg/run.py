import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch import optim

from mlmi.clustering import RandomClusterPartitioner
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import run_train_aggregate_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger


logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=False)


def run_fedavg(context: ExperimentContext, num_rounds: int):
    num_clients = 100
    steps = 10
    batch_size = 256
    learning_rate = 0.03
    optimizer_args = OptimizerArgs(optim.SGD, lr=learning_rate)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(max_steps=steps, gpus=1)
    else:
        training_args = TrainArgs(max_steps=steps)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=batch_size)

    clients = []
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client_logger = create_tensorboard_logger(context.name, str(c))
        client = FedAvgClient(str(c), model_args, context, fed_dataset.train_data_local_dict[c],
                              fed_dataset.data_local_train_num_dict[c], fed_dataset.test_data_local_dict[c],
                              fed_dataset.data_local_test_num_dict[c], client_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        clients.append(client)

    server = FedAvgServer('initial_server', model_args, context)
    num_train_samples = [client.num_train_samples for client in clients]
    for i in range(num_rounds):
        logger.info('starting training round {0}'.format(str(i + 1)))
        run_train_aggregate_round(server, clients, training_args, num_train_samples=num_train_samples)
        logger.info('finished training round')


def run_fedavg_hierarchical(context: ExperimentContext, num_rounds_init: int, num_rounds_cluster: int):
    num_clients = 100
    steps = 10
    batch_size = 256
    learning_rate = 0.03
    optimizer_args = OptimizerArgs(optim.SGD, lr=learning_rate)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(max_steps=steps, gpus=1)
    else:
        training_args = TrainArgs(max_steps=steps)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=batch_size)

    clients = []
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client_logger = create_tensorboard_logger(context.name, str(c))
        client = FedAvgClient(str(c), model_args, context, fed_dataset.train_data_local_dict[c],
                              fed_dataset.data_local_train_num_dict[c], fed_dataset.test_data_local_dict[c],
                              fed_dataset.data_local_test_num_dict[c], client_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        clients.append(client)

    server = FedAvgServer('initial_server', model_args, context)
    num_train_samples = [client.num_train_samples for client in clients]

    # Initialization of global model
    for i in range(num_rounds_init):
        logger.info('starting training round {0}'.format(str(i + 1)))
        run_train_aggregate_round(server, clients, training_args, num_train_samples=num_train_samples)
        logger.info('finished training round')

    # Clustering of participants by model updates
    partitioner = RandomClusterPartitioner()
    cluster_clients_dic = partitioner.cluster(clients)

    # Initialize cluster models
    cluster_server_dic = {}
    for cluster_id, participants in cluster_clients_dic.items():
        cluster_server_dic[cluster_id] = FedAvgServer('cluster_server'+cluster_id, model_args, context)

    # Train in clusters
    for cluster_id in cluster_clients_dic.keys():
        for i in range(num_rounds_cluster):
            logger.info('starting training cluster {1} in round {0}'.format(str(i + 1), cluster_id))
            num_train_samples = [client.num_train_samples for client in cluster_clients_dic[cluster_id]]
            run_train_aggregate_round(cluster_server_dic[cluster_id], cluster_clients_dic[cluster_id], training_args,
                                      num_train_samples=num_train_samples)
            logger.info('finished training cluster {0}'.format(cluster_id))


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        if args.hierarchical:
            context = ExperimentContext(name='fedavg_hierarchical')
            run_fedavg_hierarchical(context, 2, 2)
        else:
            context = ExperimentContext(name='fedavg_default')
            run_fedavg(context, 2)

    run()
