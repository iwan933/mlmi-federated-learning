from typing import Dict, List
import os
import random

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from mlmi.structs import FederatedDatasetData


def load_omniglot_datasets(data_dir,
                           num_clients_train: int = 1000,
                           num_clients_test: int = 200,
                           num_classes_per_client: int = 5,
                           num_shots_per_class: int = 1,
                           inner_batch_size: int = 5,
                           tensorflow: bool = False,
                           random_seed: int = None)\
        -> (FederatedDatasetData, FederatedDatasetData):
    """
    Load the Omniglot dataset.
    :param data_dir: data directory
    :param num_clients_train: number of training clients
    :param num_clients_test: number of test clients
    :param num_classes_per_client: number of omniglot characters per client
    :param num_shots_per_class: number of data samples per omniglot character per client
    :param inner_batch_size: Number of data samples per batch. A value of -1
            means batch size is equal to local training data size (full batch
            training)
    :return:
    """

    random_instance = random.Random(x=random_seed)

    omniglot_train, omniglot_test = split_dataset(
        dataset=read_dataset(data_dir, tensorflow),
        random_instance=random_instance
    )
    omniglot_train = list(augment_dataset(omniglot_train, tensorflow))
    omniglot_test = list(omniglot_test)

    omniglot_args = {
        'num_classes_per_client': num_classes_per_client,
        'num_shots_per_class': num_shots_per_class,
        'inner_batch_size': inner_batch_size
    }
    train_clients = _make_omniglot_dataset(
        dataset=omniglot_train,
        num_clients=num_clients_train,
        random_instance=random_instance,
        **omniglot_args
    )
    test_clients = _make_omniglot_dataset(
        dataset=omniglot_test,
        num_clients=num_clients_test,
        random_instance=random_instance,
        **omniglot_args
    )

    return train_clients, test_clients


def _make_omniglot_dataset(dataset,
                           num_clients: int,
                           num_classes_per_client: int = 5,
                           num_shots_per_class: int = 1,
                           inner_batch_size: int = 5,
                           random_instance = None):

    federated_dataset_args = {
        'client_num': num_clients,
        'train_data_num': 0,
        'test_data_num': 0,
        'train_data_global': None,
        'test_data_global': None,
        'data_local_num_dict': {},
        'data_local_train_num_dict': {},
        'data_local_test_num_dict': {},
        'train_data_local_dict': {},
        'test_data_local_dict': {},
        'class_num': None
    }

    for i in range(num_clients):
        client_data = list(_sample_mini_dataset(
            dataset=dataset,
            num_classes=num_classes_per_client,
            num_shots=num_shots_per_class + 1,
            random_instance=random_instance
        ))
        train_data, test_data = _split_train_test(client_data)
        train_batch_size = test_batch_size = inner_batch_size
        if inner_batch_size == -1:
            train_batch_size = len(train_data)
            test_batch_size = len(test_data)

        # Train data
        random_instance.shuffle(train_data)
        federated_dataset_args['train_data_local_dict'][i] = DataLoader(
            dataset=train_data,
            batch_size=train_batch_size,
            shuffle=False
        )
        federated_dataset_args['data_local_train_num_dict'][i] = len(train_data)
        federated_dataset_args['train_data_num'] += len(train_data)

        # Test data
        random_instance.shuffle(test_data)
        federated_dataset_args['test_data_local_dict'][i] = DataLoader(
            dataset=test_data,
            batch_size=test_batch_size,
            shuffle=False
        )
        federated_dataset_args['data_local_test_num_dict'][i] = len(test_data)
        federated_dataset_args['test_data_num'] += len(test_data)

        federated_dataset_args['data_local_num_dict'][i] = \
            len(train_data) + len(test_data)

    return FederatedDatasetData(
        name='omniglot', batch_size=inner_batch_size, **federated_dataset_args
    )


# The below code is taken from the supervised-reptile repository (code from the
# Nichol 2018 paper)
def _sample_mini_dataset(dataset, num_classes, num_shots, random_instance=None):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    if random_instance is None:
        random_instance = random.Random()
    shuffled = list(dataset)
    random_instance.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots, random_instance):
            yield (sample, class_idx)


def read_dataset(data_dir, tensorflow: bool):
    """
    Iterate over the characters in a data directory.

    Args:
      data_dir: a directory of alphabet directories.

    Returns:
      An iterable over Characters.

    The dataset is unaugmented and not split up into
    training and test sets.
    """
    for alphabet_name in sorted(os.listdir(data_dir)):
        alphabet_dir = os.path.join(data_dir, alphabet_name)
        if not os.path.isdir(alphabet_dir):
            continue
        for char_name in sorted(os.listdir(alphabet_dir)):
            if not char_name.startswith('character'):
                continue
            yield Character(os.path.join(alphabet_dir, char_name), tensorflow, 0)


def split_dataset(dataset, num_train=1200, random_instance=None):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    if random_instance is None:
        random_instance = random.Random()
    all_data = list(dataset)
    random_instance.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]

def augment_dataset(dataset, tensorflow: bool):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield Character(character.dir_path, tensorflow=tensorflow, rotation=rotation)


class Character:
    """
    A single character class.
    """

    def __init__(self, dir_path, tensorflow: bool, rotation=0):
        self.dir_path = dir_path
        self.tensorflow = tensorflow
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images, random_instance=None):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        if random_instance is None:
            random_instance = random.Random()
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random_instance.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            if self.tensorflow:
                self._cache[path] = np.array(img).astype('float32')
            else:
                self._cache[path] = np.array(img).astype('float32').reshape((1, 28, 28))
            return self._cache[path]


def _split_train_test(samples, test_shots=1):
    """
    Split a few-shot task into a train and a test set.

    Args:
      samples: an iterable of (input, label) pairs.
      test_shots: the number of examples per class in the
        test set.

    Returns:
      A tuple (train, test), where train and test are
        sequences of (input, label) pairs.
    """
    train_set = list(samples)
    test_set = []
    labels = set(item[1] for item in train_set)
    for _ in range(test_shots):
        for label in labels:
            for i, item in enumerate(train_set):
                if item[1] == label:
                    del train_set[i]
                    test_set.append(item)
                    break
    if len(test_set) < len(labels) * test_shots:
        raise IndexError('not enough examples of each class for test set')

    return train_set, test_set

