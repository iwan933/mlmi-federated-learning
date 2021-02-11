import math
import random
from typing import List, Optional, Tuple

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


def scratch_labels(fed_dataset: FederatedDatasetData, num_limit_label: int):
    """
    Scratches data to increase non-i.i.d-ness by increasing the data variety.
    :param fed_dataset: the dataset to scratch data from
    :param client_fraction_to_scratch: the fraction of clients to pick that get their data scratched
    :param fraction_to_scratch: the fraction of data per picked client to scratch
    :return:
    """
    for i in fed_dataset.train_data_local_dict.keys():
        train_dl = fed_dataset.train_data_local_dict[i]
        scratched_train_dl, train_num = scratch_labels_from_dataloaders(train_dl, fed_dataset.class_num,
                                                                        num_limit_label)
        fed_dataset.train_data_local_dict[i] = scratched_train_dl
        fed_dataset.data_local_train_num_dict[i] = train_num

        test_dl = fed_dataset.test_data_local_dict[i]
        scratched_train_dl, test_num = scratch_labels_from_dataloaders(test_dl, fed_dataset.class_num,
                                                                       num_limit_label)
        fed_dataset.test_data_local_dict[i] = scratched_train_dl
        fed_dataset.data_local_test_num_dict[i] = test_num
    return fed_dataset


def scratch_labels_from_dataloaders(
        dataloader: data.DataLoader, num_classes: int, num_limit_label: int) -> Tuple[data.DataLoader, int]:
    chosen_labels = np.random.choice(np.arange(num_classes), size=num_limit_label, replace=False)
    return _keep_only_specific_labels_from_dataloader(dataloader, chosen_labels)


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
    skip_threshold = 4
    if int(label_tensor[9:].count_nonzero()) == skip_threshold:
        # skip scratching for clients with no/only few letter labels
        num_samples = data_tensor.shape[0]
        out_dataloader = dataloader
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


def load_n_of_each_class(
        dataset: FederatedDatasetData,
        n=5,
        tabu: Optional[List[str]] = None
) -> data.DataLoader:
    client_ids = list(dataset.data_local_test_num_dict.keys())
    if tabu is not None:
        allowed_client_ids = [client_id for client_id in client_ids if client_id not in tabu]
    else:
        allowed_client_ids = client_ids

    datapoints = []
    labels = []

    for cls in range(dataset.class_num):
        for _ in range(n):
            tries = 0
            not_found = True
            while not_found:
                if tries > 30:
                    raise ValueError(f'Last 30 client picks did not have a sample of class {cls}')
                selected_client = np.random.choice(allowed_client_ids, size=1, replace=False)[0]
                test_dl = dataset.train_data_local_dict[selected_client]
                for x, y in test_dl:
                    x = x.numpy()
                    y = y.numpy()
                    if cls not in y:
                        continue
                    indices = np.argwhere(y == cls).reshape(-1)
                    index = np.random.choice(indices, size=1, replace=False)[0]
                    datapoints.append(x[index])
                    labels.append(y[index])
                    not_found = False
                    break
                tries += 1

    t_data = torch.stack([torch.from_numpy(d) for d in datapoints], dim=0)
    t_labels = torch.FloatTensor(labels)
    out_dataset = data.TensorDataset(t_data, t_labels)
    return data.DataLoader(out_dataset, batch_size=dataset.batch_size, shuffle=False, drop_last=False)


def augment_for_clustering(
        dataset: FederatedDatasetData,
        keep_as_is_percentage: float,
        cluster_num: int,
        label_core_num: int,
        label_deviation: int
):
    label_deviation_range = np.arange(-label_deviation, label_deviation + 1)
    label_array = np.arange(dataset.class_num)
    if keep_as_is_percentage != 0.0:
        keep_as_is_num = int(keep_as_is_percentage * len(dataset.train_data_local_dict.keys()))
        kept_clients = np.random.choice([k for k in list(dataset.train_data_local_dict.keys())], size=keep_as_is_num,
                                        replace=False)
    else:
        kept_clients = []

    cluster_probabilities = np.random.dirichlet(np.full((cluster_num,), 3), size=1).reshape(-1)
    cluster_labels_list = [
        np.random.choice(np.arange(dataset.class_num), size=label_core_num, replace=False) for _ in range(cluster_num)
    ]
    cluster_ids = np.arange(cluster_num)
    for key in dataset.train_data_local_dict.keys():
        if key in kept_clients:
            continue
        cluster_id = np.random.choice(cluster_ids, size=1, replace=False, p=cluster_probabilities).reshape(-1)[0]
        cluster_labels = cluster_labels_list[cluster_id]
        label_deviation_choice = np.random.choice(label_deviation_range, size=1, replace=False)
        if label_deviation_choice <= 0:
            client_labels = np.random.choice(cluster_labels, size=len(cluster_labels) + label_deviation_choice,
                                             replace=False)
        else:
            client_labels = np.random.choice([label for label in label_array if label not in cluster_labels],
                                             size=label_deviation_choice, replace=False)
            client_labels = np.append(client_labels, cluster_labels)

        dataset.train_data_local_dict[key], dataset.data_local_train_num_dict[key] = \
            _keep_only_specific_labels_from_dataloader(dataset.train_data_local_dict[key], client_labels)
        dataset.test_data_local_dict[key], dataset.data_local_test_num_dict[key] = \
            _keep_only_specific_labels_from_dataloader(dataset.test_data_local_dict[key], client_labels)


def _keep_only_specific_labels_from_dataloader(dataloader: data.DataLoader, labels_to_keep: List[int]):
    datapoints = None
    labels = None
    for x, y in dataloader:
        if datapoints is None:
            datapoints = x.numpy()
            labels = y.numpy()
            continue
        datapoints = np.concatenate((datapoints, x.numpy()))
        labels = np.concatenate((labels, y.numpy()))
    indices = np.isin(labels, labels_to_keep)

    scratched_data_tensor: Tensor = torch.from_numpy(datapoints[indices])
    scratched_label_tensor: Tensor = torch.from_numpy(labels[indices])

    dataset = data.TensorDataset(scratched_data_tensor, scratched_label_tensor)
    out_dataloader = data.DataLoader(dataset=dataset, batch_size=dataloader.batch_size, shuffle=True, drop_last=False)
    return out_dataloader, len(scratched_label_tensor)
