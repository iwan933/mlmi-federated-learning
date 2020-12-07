from fedml_api.model.cv.cnn import CNN_DropOut
from torch import optim, nn

from mlmi.fedavg.femnist import load_femnist_dataset
from mlmi.fedavg.model import FedAvgClient, FedAvgServer
from mlmi.rounds import run_train_aggregate_round
from mlmi.struct import ModelArgs, TrainArgs, CriterionArgs
from mlmi.settings import REPO_ROOT


def run_fedavg():
    num_clients = 10
    model_args = ModelArgs(CNN_DropOut, only_digits=False)
    training_args = TrainArgs(epochs=10)
    training_args.set_optimizer(optim.SGD, lr=0.03)
    criterion_args = CriterionArgs(nn.CrossEntropyLoss)
    data_dir = REPO_ROOT / 'data'
    fed_dataset = load_femnist_dataset(str(data_dir.absolute()), num_clients=num_clients, batch_size=256)

    clients = dict()
    for c, dataset in fed_dataset.train_data_local_dict.items():
        client = FedAvgClient(str(c), fed_dataset.train_data_local_dict[c], fed_dataset.data_local_train_num_dict[c],
                              fed_dataset.test_data_local_dict[c], fed_dataset.data_local_test_num_dict[c],
                              model_args, criterion_args)
        clients['fedavg_client_' + str(c)] = client

    server = FedAvgServer(model_args, training_args)

    run_train_aggregate_round(server, list(clients.values()))


if __name__ == '__main__':
    run_fedavg()
