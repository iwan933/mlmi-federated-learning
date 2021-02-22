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
from mlmi.settings import REPO_ROOT
from mlmi.structs import FederatedDatasetData, OptimizerArgs


def collate_different_sizes(batch):
    data = [item[0] for item in batch]

    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def load_ham10k_federated(
        partitions=20,
        test_size=0.15,
        batch_size=32,
        mean = (0.485, 0.456, 0.406),
        std = (0.229, 0.224, 0.225)
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
        train_data_local_dict[idx] = data.DataLoader(lazy_train_dataset, batch_size=batch_size, drop_last=False)
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
        fraction = np.random.randint(2, size=1)
        nv_label_partition_indices_ = nv_label_partition_indices_[:int(len(nv_label_partition_indices_)/fraction)-1]
        partition_indices[i] = np.concatenate((partition_indices[i], nv_label_partition_indices_)).astype(int)

    data_subsets = []
    for indices in partition_indices:
        np.random.shuffle(indices)
        data_subsets.append(ImageFolderSubset(dataset, indices.astype(int)))
    return data_subsets


def get_transformations(mean, std) -> Tuple[any, any]:
    # normalization values for pretrained resnet on Imagenet
    transform_train = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomRotation(degrees=60),
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
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    import pytorch_lightning as pl
    from torch import optim

    federated_dataset = load_ham10k_federated(partitions=27)

    dataloader = federated_dataset.train_data_local_dict[0]
    optimizer_args = OptimizerArgs(optim.SGD, lr=0.01)
    server = MobileNetV2Lightning(num_classes=7, optimizer_args=optimizer_args, participant_name='1',
                         weights=(1, 1, 1, 1, 1, 1, 1))

    models = []
    for k, dl in federated_dataset.train_data_local_dict.items():
        labels = torch.LongTensor([])
        for _, y in dl:
            labels = torch.cat((labels, y))
        counts = np.unique(labels.numpy(), return_counts=True)

        labels = torch.LongTensor([])
        for _, y in dl:
            labels = torch.cat((labels, y))
        weight = torch.ones((7,))
        label, counts = torch.unique(labels, return_counts=True)
        weight[label] = weight - counts / torch.sum(counts)

        model = MobileNetV2Lightning(num_classes=7, optimizer_args=optimizer_args, participant_name='1',
                             weights=weight)
        trainer = pl.Trainer(enable_logging=False, max_epochs=1)
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
