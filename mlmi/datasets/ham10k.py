from pathlib import Path

import pandas as pd
import numpy as np

from typing import List, Sequence, Tuple
from shutil import copyfile

from torch.utils.data import Dataset
from torch.utils import data
from torchvision import datasets
from torchvision.transforms import transforms

from mlmi.settings import REPO_ROOT
from mlmi.structs import FederatedDatasetData


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


def load_ham10k_partition_by_two_labels_federated2fulldataset(
        samples_per_package=35,
        max_samples_per_label=500,
        test_fraction=0.2,
        batch_size=8,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
) -> Tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    dataset = load_ham10k()
    train_transformations, test_transformations = get_transformations(mean, std)
    client_folder_subsets = partition_by_two_labels_per_client(
        dataset, samples_per_package, max_samples_per_label, test_fraction
    )

    train_indices = np.array([], dtype=int)
    test_indices = np.array([], dtype=int)
    for idx, (train_subset, test_subset) in enumerate(client_folder_subsets):
        train_indices = np.concatenate((train_indices, train_subset.indices))
        test_indices = np.concatenate((test_indices, test_subset.indices))

    train_subset = ImageFolderSubset(dataset, train_indices[:int(len(train_indices) * (1 - test_fraction))].astype(int))
    validation_subset = ImageFolderSubset(dataset, train_indices[int(len(train_indices) * (1 - test_fraction)):].astype(int))
    test_subset = ImageFolderSubset(dataset, test_indices.astype(int))
    train_set = LazyImageFolderDataset(train_subset[:], train_transformations)
    validation_subset = LazyImageFolderDataset(validation_subset[:], test_transformations)
    test_set = LazyImageFolderDataset(test_subset[:], test_transformations)
    train_dataloader = data.DataLoader(train_set, batch_size=batch_size, drop_last=False, shuffle=True)
    validation_dataloader = data.DataLoader(validation_subset, batch_size=batch_size, drop_last=False, shuffle=True)
    test_dataloader = data.DataLoader(test_set, batch_size=batch_size, drop_last=False)
    return train_dataloader, validation_dataloader, test_dataloader


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


def get_transformations(mean, std) -> Tuple[any, any]:
    # normalization values for pretrained resnet on Imagenet
    transform_train = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomAffine(degrees=90),
                        transforms.ColorJitter(contrast=(0.9, 1.1), saturation=(0.9, 1.1)),
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
    dataset: Dataset
    indices: Sequence[int]

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
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
