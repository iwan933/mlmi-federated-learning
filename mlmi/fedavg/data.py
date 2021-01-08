import math
import random
from typing import Tuple

import torch
import numpy as np
from torch import Tensor
from torch.utils import data

from mlmi.struct import FederatedDatasetData


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
                                          name=f'{fed_dataset.name}{n}')
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
