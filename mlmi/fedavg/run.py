import argparse

from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from pytorch_lightning.loggers import LightningLoggerBase
from torch import optim

from mlmi.clustering import RandomClusterPartitioner, GradientClusterPartitioner
from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning

from mlmi.fedavg.util import run_fedavg_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs

from mlmi.fedavg.util import run_train_aggregate_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs, ClusterArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger, evaluate_global_model


logger = getLogger(__name__)


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument('--hierarchical', dest='hierarchical', action='store_const',
                        const=True, default=True)


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


def run_fedavg(context: ExperimentContext, num_rounds: int):
    num_clients = 100
    steps = 10
    batch_size = 256
    learning_rate = 0.03
    log_every_n_steps = 3
    experiment_logger = create_tensorboard_logger(context.name,
                                                  'c{}s{}bs{}lr{}'.format(num_clients, steps, batch_size,
                                                                          str(learning_rate).replace('.', '')))
    optimizer_args = OptimizerArgs(optim.SGD, lr=learning_rate)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(max_steps=steps, log_every_n_steps=log_every_n_steps, gpus=1)
    else:
        training_args = TrainArgs(max_steps=steps, log_every_n_steps=log_every_n_steps)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=batch_size)

    clients = []
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client = FedAvgClient(str(c), model_args, context, fed_dataset.train_data_local_dict[c],
                              fed_dataset.data_local_train_num_dict[c], fed_dataset.test_data_local_dict[c],
                              fed_dataset.data_local_test_num_dict[c], experiment_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        clients.append(client)

    server = FedAvgServer('initial_server', model_args, context)
    num_train_samples = [client.num_train_samples for client in clients]
    for i in range(num_rounds):
        logger.info('starting training round {0}'.format(str(i + 1)))
        # train
        run_fedavg_round(server, clients, training_args, num_train_samples=num_train_samples)
        # test
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), experiment_logger, i)

        logger.info('finished training round')


def run_fedavg_hierarchical(context: ExperimentContext, num_rounds_init: int, num_rounds_cluster: int):
    num_clients = 4
    steps = 4
    batch_size = 20
    learning_rate = 0.03
    experiment_logger = create_tensorboard_logger(context.name,
                                                  'c{}s{}bs{}lr{}'.format(num_clients, steps, batch_size,
                                                                          str(learning_rate).replace('.', '')))
    optimizer_args = OptimizerArgs(optim.SGD, lr=learning_rate)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    cluster_args = ClusterArgs(linkage_mech='ward', dis_metric='euclidean', criterion='maxclust', max_value_criterion=4)
    if torch.cuda.is_available():
        training_args = TrainArgs(max_steps=steps, gpus=1)
    else:
        training_args = TrainArgs(max_steps=steps)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=batch_size)

    clients = []
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client = FedAvgClient(str(c), model_args, context, fed_dataset.train_data_local_dict[c],
                              fed_dataset.data_local_train_num_dict[c], fed_dataset.test_data_local_dict[c],
                              fed_dataset.data_local_test_num_dict[c], experiment_logger)
        checkpoint_callback = ModelCheckpoint(filepath=str(client.get_checkpoint_path(suffix='cb').absolute()))
        client.set_trainer_callbacks([checkpoint_callback])
        clients.append(client)

    server = FedAvgServer('initial_server', model_args, context)
    num_train_samples = [client.num_train_samples for client in clients]

    # Initialization of global model
    for i in range(num_rounds_init):
        logger.info('starting training round {0}'.format(str(i + 1)))
        # train
        run_fedavg_round(server, clients, training_args, num_train_samples=num_train_samples)
        # test
        result = evaluate_global_model(global_model_participant=server, participants=clients)
        log_loss_and_acc('global_model', result.get('test/loss'), result.get('test/acc'), experiment_logger, i)
        logger.info('finished training round')

<<<<<<< HEAD
    # Clustering of participants by model updates
    partitioner = RandomClusterPartitioner()
=======
    #Clustering
    partitioner = GradientClusterPartitioner(cluster_args)
>>>>>>> max
    cluster_clients_dic = partitioner.cluster(clients)

    # Initialize cluster models
    cluster_server_dic = {}
    for cluster_id, participants in cluster_clients_dic.items():
        cluster_server = FedAvgServer('cluster_server' + cluster_id, model_args, context)
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
            run_fedavg_round(cluster_server, cluster_clients, training_args, num_train_samples=num_train_samples)
            # test
            result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
            log_loss_and_acc('cluster{}'.format(cluster_id), result.get('test/loss'), result.get('test/acc'),
                             experiment_logger, i)

            logger.info('finished training cluster {0}'.format(cluster_id))


if __name__ == '__main__':
    def run():
        parser = argparse.ArgumentParser()
        add_args(parser)
        args = parser.parse_args()

        if args.hierarchical:
            context = ExperimentContext(name='fedavg_hierarchical')
<<<<<<< HEAD
            run_fedavg_hierarchical(context, 20, 20)
=======
            run_fedavg_hierarchical(context, 1, 2)
>>>>>>> max
        else:
            context = ExperimentContext(name='fedavg_default')
            run_fedavg(context, 80)

    run()
