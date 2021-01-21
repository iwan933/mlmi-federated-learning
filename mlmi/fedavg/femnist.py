from pathlib import Path

from fedml_api.data_preprocessing.FederatedEMNIST.data_loader import load_partition_data_federated_emnist
from torch.utils import data
from torchvision.datasets import MNIST, vision

from mlmi.structs import FederatedDatasetData

import numpy as np
from torchvision import datasets, transforms


class DatasetSplit(data.Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


def load_mnist_dataset(data_dir, num_clients=100, batch_size=10):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    trans_mnist = transforms.Compose([transforms.ToTensor()])
    dataset_train: MNIST = datasets.MNIST(data_dir, train=True, download=True, transform=trans_mnist)
    dataset_test: MNIST = datasets.MNIST(data_dir, train=False, download=True, transform=trans_mnist)

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    partitions = {i: np.array([], dtype='int64') for i in range(num_clients)}
    idxs = np.arange(num_shards*num_imgs)

    labels = dataset_train.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            partitions[i] = np.concatenate((partitions[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    test_idxs = np.split(np.arange(len(dataset_test.targets)), num_shards)

    data_local_num_dict = dict()
    data_local_test_num_dict = dict()
    data_local_train_num_dict = dict()
    train_data_local_dict = dict()
    test_data_local_dict = dict()
    for i, partition in enumerate(partitions):
        train_data_local_dict[i] = data.DataLoader(
            DatasetSplit(dataset_train, partitions[i]), batch_size=batch_size, shuffle=True)
        test_data_local_dict[i] = data.DataLoader(
            DatasetSplit(dataset_test, test_idxs[i]), batch_size=batch_size, shuffle=True)
        data_local_test_num_dict[i] = len(test_data_local_dict[i])
        data_local_train_num_dict[i] = len(train_data_local_dict[i])
        data_local_num_dict[i] = data_local_test_num_dict[i] + data_local_train_num_dict[i]

    train_data_global = data.DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=True, drop_last=False)
    test_data_global = data.DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, drop_last=False)

    result_dataset = FederatedDatasetData(client_num=num_clients,
                                          train_data_global=train_data_global,
                                          test_data_global=test_data_global,
                                          data_local_num_dict=data_local_num_dict,
                                          data_local_test_num_dict=data_local_test_num_dict,
                                          data_local_train_num_dict=data_local_train_num_dict,
                                          class_num=10,
                                          train_data_local_dict=train_data_local_dict,
                                          test_data_local_dict=test_data_local_dict,
                                          name='mnist', batch_size=batch_size)
    return result_dataset


def load_femnist_dataset(data_dir, num_clients=3400, batch_size=10) -> FederatedDatasetData:
    """
    Load the federated tensorflow emnist dataset, originally split up into 3400 clients.
    :param data_dir: data directory
    :param num_clients: number of clients to use for split
    :param batch_size: number samples per batch
    :return:
    """
    client_number, train_data_num, test_data_num, train_data_global, test_data_global, \
    data_local_num_dict, data_local_train_num_dict, data_local_test_num_dict, train_data_local_dict, \
    test_data_local_dict, class_num = load_partition_data_federated_emnist('', data_dir, client_number=num_clients,
                                                                  batch_size=batch_size)
    federated_dataset_args = (client_number, train_data_global, test_data_global,
                              data_local_num_dict, data_local_train_num_dict, data_local_test_num_dict,
                              train_data_local_dict, test_data_local_dict, class_num)
    return FederatedDatasetData(*federated_dataset_args, name='femnist', batch_size=batch_size)
