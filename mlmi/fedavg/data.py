import math
import random
from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from torch.utils import data

from mlmi.structs import FederatedDatasetData


def select_random_fed_dataset_partitions(fed_dataset: FederatedDatasetData, n: int):
    """
    Randomly select a fixed number of partitions of the federated dataset
    :param fed_dataset: the dataset to select the partitions from
    :param n: the number of partitions to select
    :return: a new federated dataset with the selected subset of partitions
    """
    indices = np.array(list(fed_dataset.train_data_local_dict.keys()))
    random_indices = np.random.choice(indices, size=n, replace=False)
    train_data_local_dict = dict()
    data_local_num_dict = dict()
    data_local_train_num_dict = dict()
    test_data_local_dict = dict()
    data_local_test_num_dict = dict()
    for i in random_indices:
        train_data_local_dict[i] = fed_dataset.train_data_local_dict[i]
        data_local_train_num_dict[i] = fed_dataset.data_local_train_num_dict[i]
        test_data_local_dict[i] = fed_dataset.test_data_local_dict[i]
        data_local_test_num_dict[i] = fed_dataset.data_local_test_num_dict[i]
        data_local_num_dict[i] = fed_dataset.data_local_num_dict[i]
    train_data_num = sum([num for num in data_local_train_num_dict.values()])
    test_data_num = sum([num for num in data_local_test_num_dict.values()])
    result_dataset = FederatedDatasetData(client_num=n, train_data_num=train_data_num, test_data_num=test_data_num,
                                          train_data_global=fed_dataset.train_data_global,
                                          test_data_global=fed_dataset.test_data_global,
                                          data_local_num_dict=data_local_num_dict,
                                          data_local_test_num_dict=data_local_test_num_dict,
                                          data_local_train_num_dict=data_local_train_num_dict,
                                          class_num=fed_dataset.class_num,
                                          train_data_local_dict=train_data_local_dict,
                                          test_data_local_dict=test_data_local_dict,
                                          name=f'{fed_dataset.name}{n}', batch_size=fed_dataset.batch_size)
    return result_dataset


def scratch_data(fed_dataset: FederatedDatasetData, client_fraction_to_scratch: float, fraction_to_scratch: float):
    """
    Scratches data to increase non-i.i.d-ness by increasing the data variety.
    :param fed_dataset: the dataset to scratch data from
    :param client_fraction_to_scratch: the fraction of clients to pick that get their data scratched
    :param fraction_to_scratch: the fraction of data per picked client to scratch
    :return:
    """
    clients_to_scratch = math.ceil(client_fraction_to_scratch * fed_dataset.client_num)
    indices = np.array(list(fed_dataset.train_data_local_dict.keys()))
    random_indices = np.random.choice(indices, size=clients_to_scratch, replace=False)
    for i in random_indices:
        train_dl = fed_dataset.train_data_local_dict[i]
        scratched_train_dl, train_num = scratch_data_from_dataloader(train_dl, fraction_to_scratch)
        fed_dataset.train_data_local_dict[i] = scratched_train_dl
        fed_dataset.data_local_train_num_dict[i] = train_num

        test_dl = fed_dataset.test_data_local_dict[i]
        scratched_train_dl, test_num = scratch_data_from_dataloader(test_dl, fraction_to_scratch)
        fed_dataset.test_data_local_dict[i] = scratched_train_dl
        fed_dataset.data_local_test_num_dict[i] = test_num


def scratch_data_from_dataloader(
        dataloader: data.DataLoader, fraction_to_scratch: float) -> Tuple[data.DataLoader, int]:
    batch_data_list = []
    batch_label_list = []
    for x, y in dataloader:
        batch_data_list.append(x)
        batch_label_list.append(y)
    data_tensor: Tensor = torch.cat(batch_data_list, 0)
    label_tensor: Tensor = torch.cat(batch_label_list, 0)
    num_samples = data_tensor.shape[0]
    num_scratched_samples = math.ceil((1 - fraction_to_scratch) * num_samples)
    indice = random.sample(range(num_samples), num_scratched_samples)
    indice = torch.tensor(indice)
    scratched_data_tensor = data_tensor[indice]
    scratched_label_tensor = label_tensor[indice]

    dataset = data.TensorDataset(scratched_data_tensor, scratched_label_tensor)
    out_dataloader = data.DataLoader(dataset=dataset, batch_size=dataloader.batch_size, shuffle=True, drop_last=False)
    return out_dataloader, num_scratched_samples


def non_iid_scratch(fed_dataset: FederatedDatasetData, num_mnist_label_zero):
    random.seed(1)
    indices_client_id = np.array(list(fed_dataset.train_data_local_dict.keys()))
    for i in indices_client_id:
        random_mnist_indices = np.array([random.randint(0, 9) for i in range(num_mnist_label_zero)])

        train_dl = fed_dataset.train_data_local_dict[i]
        scratched_mnist_data, train_num = scratch_non_idd_from_dataloader(train_dl, random_mnist_indices)
        fed_dataset.train_data_local_dict[i] = scratched_mnist_data
        fed_dataset.data_local_train_num_dict[i] = train_num

        test_dl = fed_dataset.test_data_local_dict[i]
        scratched_test_dl, test_num = scratch_non_idd_from_dataloader(test_dl, random_mnist_indices)
        fed_dataset.test_data_local_dict[i] = scratched_test_dl
        fed_dataset.data_local_test_num_dict[i] = test_num


def scratch_non_idd_from_dataloader(dataloader: data.DataLoader, random_mnist_indices):

    batch_data_list = []
    batch_label_list = []
    for x, y in dataloader:
        batch_data_list.append(x)
        batch_label_list.append(y)
    data_tensor: Tensor = torch.cat(batch_data_list, 0)
    label_tensor: Tensor = torch.cat(batch_label_list, 0)

    idx_del_list = []
    if int(label_tensor[9:].count_nonzero()) == 0:
        # skip scratching for clients with no letter labels
        pass
    else:
        # scratch random digit labels
        for index in random_mnist_indices:
            idx_del_list.append(torch.tensor([idx for idx, label in enumerate(label_tensor) if label == index]))

        idx_del_tensor: Tensor = torch.cat(idx_del_list, 0)
        for i in sorted(idx_del_tensor, reverse=True):
            i = int(i)
            data_tensor = torch.cat([data_tensor[0:i], data_tensor[i+1:]])
            label_tensor = torch.cat([label_tensor[0:i], label_tensor[i+1:]])

    dataset = data.TensorDataset(data_tensor, label_tensor)
    out_dataloader = data.DataLoader(dataset=dataset, batch_size=dataloader.batch_size, shuffle=True,
                                     drop_last=False)
    num_samples = data_tensor.shape[0]
    return out_dataloader, num_samples


def sample_data_briggs(fed_dataset: FederatedDatasetData):
    random.seed(0)
    print(fed_dataset.client_num)
    indices_client_id = np.array(list(fed_dataset.train_data_local_dict.keys()))
    for i in indices_client_id:
        num_data_points = random.randint(12, 386)
        train_dl = fed_dataset.train_data_local_dict[i]
        sampled_data_briggs = scratch_data_briggs(train_dl, num_data_points)
        fed_dataset.train_data_local_dict[i] = sampled_data_briggs
        fed_dataset.data_local_train_num_dict[i] = num_data_points

        test_dl = fed_dataset.test_data_local_dict[i]
        sampled_data_briggs = scratch_data_briggs(test_dl, num_data_points)
        fed_dataset.test_data_local_dict[i] = sampled_data_briggs
        fed_dataset.data_local_test_num_dict[i] = num_data_points


def scratch_data_briggs(dataloader: data.DataLoader, num_data_points: int):
    batch_data_list = []
    batch_label_list = []
    for x, y in dataloader:
        batch_data_list.append(x)
        batch_label_list.append(y)
    data_tensor: Tensor = torch.cat(batch_data_list, 0)
    label_tensor: Tensor = torch.cat(batch_label_list, 0)
    num_samples = data_tensor.shape[0]
    indices = random.sample(range(num_samples), num_data_points)
    indices = torch.tensor(indices)
    scratched_data_tensor = data_tensor[indices]
    scratched_label_tensor = label_tensor[indices]

    dataset = data.TensorDataset(scratched_data_tensor, scratched_label_tensor)
    out_dataloader = data.DataLoader(dataset=dataset, batch_size=dataloader.batch_size, shuffle=True,
                                     drop_last=False)
    return out_dataloader




