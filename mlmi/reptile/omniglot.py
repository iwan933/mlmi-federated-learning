from typing import Dict, List
import os
import random

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader

from mlmi.reptile.omniglot import read_dataset, split_dataset
from mlmi.struct import FederatedDatasetData


def load_omniglot_dataset(data_dir,
                          num_clients_train: int = 1000,
                          num_clients_test: int = 200,
                          num_classes_per_client: int = 5,
                          num_shots_per_class: int = 1,
                          inner_batch_size: int = 5) -> FederatedDatasetData:
    """
    Load the Omniglot dataset.
    :param data_dir: data directory
    :param num_clients_train: number of training clients
    :param num_clients_test: number of test clients
    :param num_classes_per_client: number of omniglot characters per client
    :param num_shots_per_class: number of data samples per omniglot character per client
    :param inner_batch_size: number of data samples per batch
    :return:
    """

    omniglot_train, omniglot_test = split_dataset(read_dataset(data_dir))
    omniglot_train = list(augment_dataset(omniglot_train))
    omniglot_test = list(omniglot_test)

    federated_dataset_args = {}
    federated_dataset_args['client_num'] = num_clients_train
    federated_dataset_args['train_data_num'] = 0
    federated_dataset_args['test_data_num'] = 0
    federated_dataset_args['train_data_global'] = None
    federated_dataset_args['test_data_global'] = None
    federated_dataset_args['data_local_num_dict'] = {}
    federated_dataset_args['data_local_train_num_dict'] = {}
    federated_dataset_args['data_local_test_num_dict'] = {}
    federated_dataset_args['train_data_local_dict'] = {}
    federated_dataset_args['test_data_local_dict'] = {}
    federated_dataset_args['class_num'] = None

    for i in range(num_clients_train):
        client_train_data = list(_sample_mini_dataset(
            dataset=omniglot_train,
            num_classes=num_classes_per_client,
            num_shots=num_shots_per_class
        ))
        federated_dataset_args['train_data_local_dict'][i] = DataLoader(
            dataset=client_train_data,
            batch_size=inner_batch_size,
            shuffle=True
        )
        federated_dataset_args['data_local_train_num_dict'][i] = len(client_train_data)
        federated_dataset_args['train_data_num'] += len(client_train_data)
        federated_dataset_args['data_local_num_dict'][i] = len(client_train_data)

    for i in range(num_clients_test):
        client_test_data = _sample_mini_dataset(
            dataset=omniglot_test,
            num_classes=num_classes_per_client,
            num_shots=num_shots_per_class + 1
        )
        # Split test data into test-train and test-test set. At test time, train
        # model on test-train data and test on test-test data of unseen test clients.
        test_train_set, test_test_set = _split_train_test(client_test_data)
        # 'test_data_local_dict' will contain a tuple of dataloader, test_train_set,
        # and test_test_set
        federated_dataset_args['test_data_local_dict'][i] = (
            DataLoader(
                dataset=test_train_set,
                batch_size=inner_batch_size,
                shuffle=True
            ),
            test_train_set,
            test_test_set,
        )
        federated_dataset_args['data_local_test_num_dict'][i] = len(test_train_set)
        federated_dataset_args['test_data_num'] += len(test_train_set)

    return FederatedDatasetData(**federated_dataset_args)


def _sample_mini_dataset(dataset, num_classes, num_shots):
    """
    Sample a few shot task from a dataset.

    Returns:
      An iterable of (input, label) pairs.
    """
    shuffled = list(dataset)
    random.shuffle(shuffled)
    for class_idx, class_obj in enumerate(shuffled[:num_classes]):
        for sample in class_obj.sample(num_shots):
            yield (sample, class_idx)


def read_dataset(data_dir):
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
            yield Character(os.path.join(alphabet_dir, char_name), 0)


def split_dataset(dataset, num_train=1200):
    """
    Split the dataset into a training and test set.

    Args:
      dataset: an iterable of Characters.

    Returns:
      A tuple (train, test) of Character sequences.
    """
    all_data = list(dataset)
    random.shuffle(all_data)
    return all_data[:num_train], all_data[num_train:]


def augment_dataset(dataset):
    """
    Augment the dataset by adding 90 degree rotations.

    Args:
      dataset: an iterable of Characters.

    Returns:
      An iterable of augmented Characters.
    """
    for character in dataset:
        for rotation in [0, 90, 180, 270]:
            yield Character(character.dir_path, rotation=rotation)


# pylint: disable=R0903
class Character:
    """
    A single character class.
    """

    def __init__(self, dir_path, rotation=0):
        self.dir_path = dir_path
        self.rotation = rotation
        self._cache = {}

    def sample(self, num_images):
        """
        Sample images (as numpy arrays) from the class.

        Returns:
          A sequence of 28x28 numpy arrays.
          Each pixel ranges from 0 to 1.
        """
        names = [f for f in os.listdir(self.dir_path) if f.endswith('.png')]
        random.shuffle(names)
        images = []
        for name in names[:num_images]:
            images.append(self._read_image(os.path.join(self.dir_path, name)))
        return images

    def _read_image(self, path):
        if path in self._cache:
            return self._cache[path]
        with open(path, 'rb') as in_file:
            img = Image.open(in_file).resize((28, 28)).rotate(self.rotation)
            self._cache[path] = np.array(img).astype('float32')
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