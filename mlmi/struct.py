from typing import Type, Dict
from torch import optim
from torch.utils import data
import pytorch_lightning as pl

<<<<<<< HEAD
=======


>>>>>>> implementations from programming session

class ExperimentContext(object):
    """
    Structure to hold experiment context information
    """
    def __init__(self, name: str):
        self.name = name


class TrainArgs(object):
    """
    Structure to hold arguments used to be passed to training instances. Arguments should all be
    serializable in case the arguments are send over wire.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class OptimizerArgs(object):
    """
    Arguments for optimizer construction
    """
    def __init__(self, optimizer_class: Type[optim.Optimizer], *args, **kwargs):
        self.optimizer_class = optimizer_class
        self.optimizer_args = args
        self.optimizer_kwargs = kwargs


class ModelArgs(object):
    """
    Arguments for model construction
    """
    def __init__(self, model_class: Type[pl.LightningModule], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model_class = model_class


class FederatedDatasetData(object):
    """
    Dataset type resulting from FedML dataset loader
    """
    def __init__(self, client_num, train_data_num: int, test_data_num: int, train_data_global: data.DataLoader,
                 test_data_global: data.DataLoader, data_local_num_dict: Dict[int, int],
                 data_local_train_num_dict: Dict[int, int], data_local_test_num_dict: Dict[int, int],
                 train_data_local_dict: Dict[int, data.DataLoader], test_data_local_dict: Dict[int, data.DataLoader],
                 class_num: int):
        self.client_num = client_num
        self.train_data_num = train_data_num
        self.test_data_num = test_data_num
        self.train_data_global = train_data_global
        self.test_data_global = test_data_global
        self.data_local_num_dict = data_local_num_dict
        self.data_local_train_num_dict = data_local_train_num_dict
        self.data_local_test_num_dict = data_local_test_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.class_num = class_num
