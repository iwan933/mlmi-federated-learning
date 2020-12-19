from pytorch_lightning.callbacks import ModelCheckpoint

import torch
from torch import optim

from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer, CNNLightning
from mlmi.fedavg.util import run_train_aggregate_round
from mlmi.struct import ExperimentContext, ModelArgs, TrainArgs, OptimizerArgs
from mlmi.settings import REPO_ROOT
from mlmi.utils import create_tensorboard_logger


def run_fedavg(context: ExperimentContext):
    num_clients = 100
    epochs = 2
    optimizer_args = OptimizerArgs(optim.SGD, lr=0.03)
    model_args = ModelArgs(CNNLightning, optimizer_args, only_digits=False)
    if torch.cuda.is_available():
        training_args = TrainArgs(epochs=2, gpus=1)
    else:
        training_args = TrainArgs(epochs=2)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=256)

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
    run_train_aggregate_round(server, clients, training_args, num_train_samples=num_train_samples)


if __name__ == '__main__':
    def run():
        context = ExperimentContext(name='fedavg_default')
        run_fedavg(context)

    run()
