from typing import Optional, Type, Dict
from torch import optim
from torch.utils import data
import pytorch_lightning as pl


class TrainArgs(object):
    """
    Structure to hold arguments used to be passed to training instances. Arguments should all be
    serializable in case the arguments are send over wire.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class ClusterArgs(object):
    """
    Arguments for clustering
    """
    def __init__(self, partitioner_class: Type['BaseClusterPartitioner'], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.partitioner_class = partitioner_class
        self._config_string = self._create_config_string(**kwargs)

    def _create_config_string(self, **kwargs):
        unique_str = ''
        linkage_mech = kwargs.get('linkage_mech', None)
        criterion = kwargs.get('criterion', None)
        dis_metric = kwargs.get('dis_metric', None)
        max_value_criterion = kwargs.get('max_value_criterion', None)
        if linkage_mech == 'ward':
            unique_str += 'w'
        else:
            raise ValueError(f'No shortform of linkage_mech "{linkage_mech}" known')
        if criterion == 'distance':
            unique_str += '_dist'
        else:
            raise ValueError(f'No shortform of criterion "{criterion}" known')
        if dis_metric == 'euclidean':
            unique_str += '_eu'
        else:
            raise ValueError(f'No shortform of dis_metric "{dis_metric}" known')
        unique_str += f'{max_value_criterion:.2f}'
        return unique_str

    def __str__(self):
        return self._config_string


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
                 class_num: int, name: str):
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
        self.name = name


class ExperimentContext(object):
    """
    Structure to hold experiment context information
    """

    def __init__(self, name: str, client_fraction: float, local_epochs: int, lr: float,
                 batch_size: int, optimizer_args: OptimizerArgs, train_args: TrainArgs, model_args: ModelArgs,
                 dataset: FederatedDatasetData, cluster_args: Optional['ClusterArgs'] = None):
        self.name = name
        self.client_fraction = client_fraction
        self.local_epochs = local_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.optimizer_args = optimizer_args
        self.train_args = train_args
        self.model_args = model_args
        self.cluster_args = cluster_args
        self.dataset = dataset
        self._experiment_logger = None

    @property
    def experiment_logger(self):
        return self._experiment_logger

    @experiment_logger.setter
    def experiment_logger(self, value):
        self._experiment_logger = value

    def __str__(self):
        """
        String identifying experiment. Used for model loading and saving.
        :return:
        """
        id = f'{self.dataset.name}_bs{self.batch_size}lr{self.lr:.2E}cf{self.client_fraction:.2f}e{self.local_epochs}'
        return id
