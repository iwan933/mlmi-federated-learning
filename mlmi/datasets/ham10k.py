from pathlib import Path

import pandas as pd
import numpy as np
import torch

from typing import List, Optional, Sequence, T_co, Tuple
from shutil import copyfile

from torch.utils.data import Dataset, SubsetRandomSampler
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms
import torchvision.models as models

from mlmi.models.ham10k import MobileNetV2Lightning
from mlmi.plot import generate_data_label_heatmap
from mlmi.settings import REPO_ROOT
from mlmi.structs import FederatedDatasetData, OptimizerArgs
from mlmi.utils import create_tensorboard_logger


def collate_different_sizes(batch):
    data = [item[0] for item in batch]

    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def load_ham10k_federated(
        partitions=20,
        test_size=0.15,
        batch_size=32,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
) -> FederatedDatasetData:
    dataset = load_ham10k()
    subsets = partition_ham10k_dataset(dataset, partitions=partitions)
    train_transformations, test_transformations = get_transformations(mean, std)
    lazy_datasets = []
    for subset in subsets:
        test_split = int(len(subset) * (1 - test_size))
        train_set = LazyImageFolderDataset(subset[:test_split], train_transformations)
        test_set = LazyImageFolderDataset(subset[test_split:], test_transformations)
        lazy_datasets.append((train_set, test_set))

    data_local_test_num_dict = {}
    data_local_train_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for idx, (lazy_train_dataset, lazy_test_dataset) in enumerate(lazy_datasets):
        train_data_local_dict[idx] = data.DataLoader(lazy_train_dataset, batch_size=batch_size, drop_last=False,
                                                     shuffle=True)
        test_data_local_dict[idx] = data.DataLoader(lazy_test_dataset, batch_size=batch_size, drop_last=False)
        data_local_train_num_dict[idx] = len(lazy_train_dataset)
        data_local_test_num_dict[idx] = len(lazy_test_dataset)
    return FederatedDatasetData(
        client_num=partitions,
        train_data_global=None,
        test_data_global=None,
        data_local_num_dict=None,
        data_local_test_num_dict=data_local_test_num_dict,
        data_local_train_num_dict=data_local_train_num_dict,
        class_num=7,
        train_data_local_dict=train_data_local_dict,
        test_data_local_dict=test_data_local_dict,
        name='ham10k',
        batch_size=batch_size
    )


def load_ham10k_few_big_many_small_federated(
        num_big_clients=20,
        num_small_clients=80,
        num_small_tripples_per_client=10,
        test_fraction=0.2,
        batch_size=32,
        min_big_tripple=75,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    dataset = load_ham10k()
    train_client_data, test_client_data = _load_ham10k_few_big_many_small_federated(dataset, num_big_clients,
                                                                                    num_small_clients,
                                                                                    num_small_tripples_per_client,
                                                                                    test_fraction, batch_size,
                                                                                    min_big_tripple, mean, std)
    return _create_federated_dataloader(train_client_data, test_client_data, batch_size)


def _load_ham10k_few_big_many_small_federated(
        dataset,
        num_big_clients=20,
        num_small_clients=80,
        num_small_tripples_per_client=10,
        test_fraction=0.2,
        batch_size=32,
        min_big_tripple=75,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
) -> Tuple[List[Tuple['LazyImageFolderDataset', 'LazyImageFolderDataset']], List[Tuple['LazyImageFolderDataset', 'LazyImageFolderDataset']]]:
    large_train_subsets, large_test_subsets, small_train_subsets, small_test_subsets = \
        partition_ham10k_few_big_many_small_dataset(dataset, test_fraction=test_fraction,
                                                                              num_big_clients=num_big_clients,
                                                                              num_small_clients=num_small_clients,
                                                                              num_small_tripples_per_client=
                                                                              num_small_tripples_per_client,
                                                                              min_big_tripple=min_big_tripple)
    train_transformations, test_transformations = get_transformations(mean, std)
    large_lazy_datasets = []
    for train_subset, test_subset in zip(large_train_subsets, large_test_subsets):
        train_set = LazyImageFolderDataset(train_subset, train_transformations)
        test_set = LazyImageFolderDataset(test_subset, test_transformations)
        large_lazy_datasets.append((train_set, test_set))

    small_lazy_datasets = []
    for train_subset, test_subset in zip(small_train_subsets, small_test_subsets):
        train_set = LazyImageFolderDataset(train_subset, train_transformations)
        test_set = LazyImageFolderDataset(test_subset, test_transformations)
        small_lazy_datasets.append((train_set, test_set))

    all_indices = np.arange(len(small_lazy_datasets))
    test_indices = np.random.choice(all_indices, size=40, replace=False)
    train_indices = np.delete(all_indices, test_indices)

    train_datasets = large_lazy_datasets

    for train_index in train_indices:
        train_datasets.append(small_lazy_datasets[train_index])

    test_datasets = []
    for test_index in test_indices:
        test_datasets.append(small_lazy_datasets[test_index])
    return train_datasets, test_datasets


def _create_federated_dataloader(train_datasets, test_datasets, batch_size):
    data_local_test_num_dict = {}
    data_local_train_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for idx, (lazy_train_dataset, lazy_test_dataset) in enumerate(train_datasets):
        train_data_local_dict[idx] = data.DataLoader(lazy_train_dataset, batch_size=batch_size, drop_last=False,
                                                     shuffle=True)
        test_data_local_dict[idx] = data.DataLoader(lazy_test_dataset, batch_size=batch_size, drop_last=False)
        data_local_train_num_dict[idx] = len(lazy_train_dataset)
        data_local_test_num_dict[idx] = len(lazy_test_dataset)

    test_data_local_test_num_dict = {}
    test_data_local_train_num_dict = {}
    test_train_data_local_dict = {}
    test_test_data_local_dict = {}

    for idx in range(len(test_datasets)):
        lazy_train_dataset, lazy_test_dataset = test_datasets[idx]
        test_train_data_local_dict[idx] = data.DataLoader(lazy_train_dataset, batch_size=batch_size, drop_last=False,
                                                          shuffle=True)
        test_test_data_local_dict[idx] = data.DataLoader(lazy_test_dataset, batch_size=batch_size, drop_last=False)
        test_data_local_train_num_dict[idx] = len(lazy_train_dataset)
        test_data_local_test_num_dict[idx] = len(lazy_test_dataset)

    train_set_dataset = FederatedDatasetData(
        client_num=len(train_datasets),
        train_data_global=None,
        test_data_global=None,
        data_local_num_dict=None,
        data_local_test_num_dict=data_local_test_num_dict,
        data_local_train_num_dict=data_local_train_num_dict,
        class_num=7,
        train_data_local_dict=train_data_local_dict,
        test_data_local_dict=test_data_local_dict,
        name='ham10k',
        batch_size=batch_size
    )

    test_set_dataset = FederatedDatasetData(
        client_num=len(test_datasets),
        train_data_global=None,
        test_data_global=None,
        data_local_num_dict=None,
        data_local_test_num_dict=test_data_local_test_num_dict,
        data_local_train_num_dict=test_data_local_train_num_dict,
        class_num=7,
        train_data_local_dict=test_train_data_local_dict,
        test_data_local_dict=test_test_data_local_dict,
        name='ham10k',
        batch_size=batch_size
    )

    return train_set_dataset, test_set_dataset


def load_ham10k_few_big_many_small_federated2fulldataset(
        num_big_clients=20,
        num_small_clients=80,
        num_small_tripples_per_client=10,
        test_fraction=0.2,
        batch_size=8,
        min_big_tripple=75,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
):
    dataset = load_ham10k()
    train_transformations, test_transformations = get_transformations(mean, std)
    train_client_data, test_client_data = _load_ham10k_few_big_many_small_federated(
        dataset,
        num_big_clients=num_big_clients,
        num_small_clients=num_small_clients,
        num_small_tripples_per_client=num_small_tripples_per_client,
        test_fraction=test_fraction,
        batch_size=batch_size,
        min_big_tripple=min_big_tripple,
        mean=mean,
        std=std
    )
    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)
    datasets: List[Tuple['LazyImageFolderDataset', 'LazyImageFolderDataset']] = [*train_client_data, *test_client_data]
    for train_dataset, test_dataset in datasets:
        train_indices = np.concatenate((train_indices, train_dataset.dataset.indices))
        test_indices = np.concatenate((test_indices, test_dataset.dataset.indices))

    train_subset = ImageFolderSubset(dataset, train_indices.astype(int))
    test_subset = ImageFolderSubset(dataset, test_indices.astype(int))
    train_set = LazyImageFolderDataset(train_subset[:], train_transformations)
    test_set = LazyImageFolderDataset(test_subset[:], test_transformations)
    train_dataloader = data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True)
    test_dataloader = data.DataLoader(test_set, batch_size=batch_size, drop_last=False)
    return train_dataloader, test_dataloader


def load_ham10k_partition_by_two_labels_federated(
        samples_per_package=35,
        max_samples_per_label=500,
        test_fraction=0.2,
        batch_size=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
) -> FederatedDatasetData:
    dataset = load_ham10k()
    train_transformations, test_transformations = get_transformations(mean, std)
    client_folder_subsets = partition_by_two_labels_per_client(
        dataset, samples_per_package, max_samples_per_label, test_fraction
    )

    data_local_test_num_dict = {}
    data_local_train_num_dict = {}
    train_data_local_dict = {}
    test_data_local_dict = {}

    for idx, (train_subset, test_subset) in enumerate(client_folder_subsets):
        train_set = LazyImageFolderDataset(train_subset, train_transformations)
        test_set = LazyImageFolderDataset(test_subset, test_transformations)

        train_data_local_dict[idx] = data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True)
        test_data_local_dict[idx] = data.DataLoader(test_set, batch_size=batch_size, drop_last=False)
        data_local_train_num_dict[idx] = len(train_set)
        data_local_test_num_dict[idx] = len(test_set)

    return FederatedDatasetData(
        client_num=len(client_folder_subsets),
        train_data_global=None,
        test_data_global=None,
        data_local_num_dict=None,
        data_local_test_num_dict=data_local_test_num_dict,
        data_local_train_num_dict=data_local_train_num_dict,
        class_num=7,
        train_data_local_dict=train_data_local_dict,
        test_data_local_dict=test_data_local_dict,
        name='ham10k',
        batch_size=batch_size
    )


def load_ham10k() -> 'datasets.ImageFolder':
    data_dir = REPO_ROOT / 'data'
    ham10k_dir = data_dir / 'ham10k'
    image_dir = ham10k_dir / 'images'
    if not image_dir.exists():
        prepare_image_copy(ham10k_dir)
    dataset = datasets.ImageFolder(str(image_dir.absolute()))
    return dataset


def prepare_image_copy(ham10k_dir: Path):
    image_dir = ham10k_dir / 'images'
    ham10k_metafile = ham10k_dir / 'HAM10000_metadata.csv'
    ham10k_images_part1 = ham10k_dir / 'HAM10000_images_part_1'
    ham10k_images_part2 = ham10k_dir / 'HAM10000_images_part_2'
    df = pd.read_csv(str(ham10k_metafile.absolute()), sep=',')
    labels = df['dx'].unique()
    for label in labels:
        (image_dir / label).mkdir(parents=True, exist_ok=True)
        label_datarows = df[df['dx'] == label]
        for image_id in label_datarows['image_id']:
            image_filename = f'{image_id}.jpg'
            for img_directory in [ham10k_images_part1, ham10k_images_part2]:
                if (img_directory / image_filename).exists():
                    copyfile(str((img_directory / image_filename).absolute()),
                             str((image_dir / label / image_filename).absolute()))


def partition_ham10k_dataset(
        dataset: 'datasets.ImageFolder',
        partitions: Optional[int] = None
) -> List['ImageFolderSubset']:
    """
    Partitions the ham 10k dataset. Each partition is generated in the following way,
    1. each label except for the 'nv-label' are split into 10 parts
    2. depending on the number of partitions each partition gets assigned (labels*10-1)/partitions parts
    3. for each label except for 'nv-label' the remaining part is split among the partitions
    4. the nv-label is distributed evenly accross the partitions
    :param dataset:
    :param distribution_probabilities:
    :return:
    """
    unique_label, unique_inverse, unique_counts = np.unique(dataset.targets,
                                                            return_inverse=True, return_counts=True)
    num_classes = len(unique_label)
    max_partitions = int((num_classes - 1) * 9)
    if partitions is None:
        partitions = max_partitions

    nv_label = unique_label[np.argmax(unique_counts)]
    other_labels = unique_label[unique_label != nv_label]
    label_parts_indices = []
    label_split_parts = []
    for label in other_labels:
        label_indices = np.array(np.where(unique_inverse == label), dtype=int).reshape(-1)
        label_indices = label_indices[:int(len(label_indices)/10)*10]
        label_indices = label_indices.reshape((10, -1))
        for part in label_indices[:-1]:
            label_parts_indices.append(part)
        label_split_parts.append(label_indices[-1])
    label_per_partition = int(1 / (partitions / max_partitions))
    partition_indices = np.random.choice(label_parts_indices, size=(partitions, label_per_partition), replace=False)
    partition_indices = [np.concatenate(indices) for indices in partition_indices]
    for label_split_part in label_split_parts:
        split_partition_indices = np.random.choice(label_split_part,
                                                   size=(partitions, int(len(label_split_part) / partitions)),
                                                   replace=False)
        for i in range(partitions):
            partition_indices[i] = np.concatenate((partition_indices[i], split_partition_indices[i]))
    nv_label_indices = np.array(np.where(unique_inverse == nv_label), dtype=int).reshape(-1)
    nv_label_partition_indices = np.random.choice(nv_label_indices,
                                                  size=(partitions, int(len(nv_label_indices) / partitions)),
                                                  replace=False)
    for i in range(partitions):
        nv_label_partition_indices_ = nv_label_partition_indices[i]
        fraction = np.random.randint(1, 5, size=1)
        nv_label_partition_indices_ = nv_label_partition_indices_[:int(len(nv_label_partition_indices_)/fraction)-1]
        partition_indices[i] = np.concatenate((partition_indices[i], nv_label_partition_indices_)).astype(int)

    data_subsets = []
    for indices in partition_indices:
        np.random.shuffle(indices)
        data_subsets.append(ImageFolderSubset(dataset, indices.astype(int)))
    return data_subsets


def partition_by_two_labels_per_client(
        dataset: 'datasets.ImageFolder',
        samples_per_package: int = 40,
        max_samples_per_label: int = 500,
        test_fraction: float = 0.2
) -> List[Tuple['ImageFolderSubset', 'ImageFolderSubset']]:
    unique_label, unique_inverse, unique_counts = np.unique(dataset.targets,
                                                            return_inverse=True, return_counts=True)
    for label in unique_label:
        label_indices = np.where(unique_inverse == label)[0]
        np.random.shuffle(label_indices)
        # invalidate label to perform downsampling
        unique_inverse[label_indices[min(max_samples_per_label, len(label_indices)):]] = -1
    client_indices_list = []
    while len(unique_label) >= 2:
        chosen_labels = np.random.choice(unique_label, size=2, replace=False)
        client_label_indices = np.array([], dtype=int)
        for label in chosen_labels:
            label_indices = np.where(unique_inverse == label)[0]
            chosen_label_indices = np.random.choice(label_indices, size=samples_per_package, replace=False)
            unique_inverse[chosen_label_indices] = -1  # invalidate the used indices by changing the label
            client_label_indices = np.append(client_label_indices, chosen_label_indices)
            if len(np.where(unique_inverse == label)[0]) < samples_per_package:
                unique_label = np.delete(unique_label, np.where(unique_label == label)[0])
        client_indices_list.append(client_label_indices)
    client_folder_subsets = []
    for client_indices in client_indices_list:
        np.random.shuffle(client_indices)
        train_indices = client_indices[:int(len(client_indices) * (1 - test_fraction))]
        test_indices = client_indices[int(len(client_indices) * (1 - test_fraction)):]
        train_subset = ImageFolderSubset(dataset, train_indices.astype(int))
        test_subset = ImageFolderSubset(dataset, test_indices.astype(int))
        client_folder_subsets.append((train_subset, test_subset))
    return client_folder_subsets


def partition_ham10k_few_big_many_small_dataset(
        dataset: 'datasets.ImageFolder',
        num_big_clients,
        num_small_clients,
        num_small_tripples_per_client,
        test_fraction,
        min_big_tripple
) -> Tuple[List['ImageFolderSubset'], List['ImageFolderSubset'], List['ImageFolderSubset'], List['ImageFolderSubset']]:
    small_train_data_subsets = []
    small_test_data_subsets = []
    large_train_data_subsets = []
    large_test_data_subsets = []

    unique_label, unique_inverse, unique_counts = np.unique(dataset.targets,
                                                            return_inverse=True, return_counts=True)

    tripples_list = []
    for label in unique_label:
        label_indices = np.where(unique_inverse == label)[0]
        label_indices = label_indices[:(len(label_indices) - (len(label_indices) % 3))]
        label_tripples = label_indices.reshape((-1, 3))
        tripples_list.append(label_tripples)
    tripples = np.concatenate(tripples_list, axis=0)

    small_choices = np.random.choice(np.arange(len(tripples)), size=(num_small_clients, num_small_tripples_per_client),
                                     replace=False)
    for choices in small_choices:
        train_indices = np.transpose(np.transpose(tripples[choices])[:2]).reshape(-1)
        test_indices = np.transpose(np.transpose(tripples[choices])[2:]).reshape(-1)
        small_train_data_subsets.append(ImageFolderSubset(dataset, train_indices.astype(int)))
        small_test_data_subsets.append(ImageFolderSubset(dataset, test_indices.astype(int)))
    remaining_tripples = np.delete(tripples, small_choices.reshape(-1), axis=0)

    for i in range(num_big_clients):
        max_big_tripple = len(remaining_tripples) / (num_big_clients - i)
        num_choices = np.random.randint(min_big_tripple, max_big_tripple + 1)
        choices = np.random.choice(np.arange(len(remaining_tripples)), size=num_choices, replace=False)
        selected_indices = remaining_tripples[choices].reshape(-1)
        np.random.shuffle(selected_indices)
        train_indices = selected_indices[:int(len(selected_indices) * (1 - test_fraction))]
        test_indices = selected_indices[int(len(selected_indices) * (1 - test_fraction)):]
        remaining_tripples = np.delete(remaining_tripples, choices, axis=0)
        large_train_data_subsets.append(ImageFolderSubset(dataset, train_indices.astype(int)))
        large_test_data_subsets.append(ImageFolderSubset(dataset, test_indices.astype(int)))

    return large_train_data_subsets, large_test_data_subsets, small_train_data_subsets, small_test_data_subsets


def get_transformations(mean, std) -> Tuple[any, any]:
    # normalization values for pretrained resnet on Imagenet
    transform_train = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(degrees=60, scale=(1.0, 2.0)),
                        #transforms.RandomApply([transforms.ColorJitter(brightness=(0.7, 1.3))], p=0.3),
                        #transforms.RandomApply([transforms.ColorJitter(contrast=(0.7, 1.3))], p=0.3),
                        #transforms.RandomApply([transforms.ColorJitter(saturation=(0.7, 1.3))], p=0.3),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        ])
    transform_test = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean, std),
                        ])
    return transform_train, transform_test


class ImageFolderSubset(Dataset):
    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        temp = [self.dataset[i] for i in self.indices[idx]]
        return temp

    def __len__(self):
        return len(self.indices)


class LazyImageFolderDataset(Dataset):

    def __init__(self, dataset: ImageFolderSubset, transform=None):
        self.dataset = dataset
        self.data = self.dataset[:]
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.data[index][0], self.data[index][1]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    experiment_logger = create_tensorboard_logger('transform_test')

    from torch import optim

    optimizer_args = OptimizerArgs(optim.SGD, lr=0.01)
    model = MobileNetV2Lightning(num_classes=7, optimizer_args=optimizer_args, participant_name='1',
                                  weights=torch.FloatTensor([1, 1, 1, 1, 1, 1, 1]))
    train_transform, test_transform = get_transformations(mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225))
    print(model)
    dataset = load_ham10k()
    x, y = dataset[0]
    for i in range(20):
        _x = train_transform(x)
        experiment_logger.experiment.add_image(f'{i}', _x)

    _fed_dataset = load_ham10k_partition_by_two_labels_federated()
    tag = 'ham10k2label'
    experiment_logger = create_tensorboard_logger('datadistribution', _fed_dataset.name)
    dataloaders = list(_fed_dataset.train_data_local_dict.values())
    image = generate_data_label_heatmap(tag, dataloaders, _fed_dataset.class_num)
    experiment_logger.experiment.add_image(f'label distribution/{tag}', image.numpy())
    print('')
    """
    import pytorch_lightning as pl
    from torch import optim

    federated_dataset = load_ham10k_federated(partitions=27)

    dataloader = federated_dataset.train_data_local_dict[0]
    optimizer_args = OptimizerArgs(optim.SGD, lr=0.01)
    server = MobileNetV2Lightning(num_classes=7, optimizer_args=optimizer_args, participant_name='1',
                         weights=torch.FloatTensor([1, 1, 1, 1, 1, 1, 1]))

    models = []
    for k, dl in list(federated_dataset.train_data_local_dict.items())[:3]:
        labels = torch.LongTensor([])
        for _, y in dl:
            labels = torch.cat((labels, y))
        counts = np.unique(labels.numpy(), return_counts=True)

        labels = torch.LongTensor([])
        for _, y in dl:
            labels = torch.cat((labels, y))
        weight = torch.ones((7,))
        label, counts = torch.unique(labels, return_counts=True)
        weight[label] = weight[label] - counts / torch.sum(counts)

        model = MobileNetV2Lightning(num_classes=7, optimizer_args=optimizer_args, participant_name='1',
                             weights=weight)
        trainer = pl.Trainer(max_epochs=3)
        trainer.fit(model, dl)
        models.append(model)

    from mlmi.clustering import flatten_model_parameter

    model_states = [m.state_dict() for m in models]
    keys = list(model_states[0].keys())
    model_parameter = np.array([flatten_model_parameter(m, keys).numpy() for m in model_states], dtype=float)

    global_parameter = flatten_model_parameter(server.state_dict(), keys).cpu().numpy()
    euclidean_dist = np.array([((model_parameter[participant_id] - global_parameter) ** 2).sum(axis=0)
                               for participant_id in range(len(models))])
    print(euclidean_dist)
    """
