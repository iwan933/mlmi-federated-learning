from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch import optim

from mlmi.log import getLogger
from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import run_train_aggregate_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger


logger = getLogger(__name__)


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
    num_train_samples = [client.get_num_train_samples() for client in clients]
    for i in range(num_rounds):
        logger.info('starting training round {0}'.format(str(i + 1)))
        run_train_aggregate_round(server, clients, training_args, num_train_samples=num_train_samples)
        logger.info('finished training round')


if __name__ == '__main__':
    def run():
        context = ExperimentContext(name='fedavg_default')
        run_fedavg(context, 2)

    run()
