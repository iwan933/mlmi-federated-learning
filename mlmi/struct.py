from typing import Type, Dict
from torch import optim, nn
from torch.utils import data


class TrainResults(object):
    model_state = dict()
    loss_history = []
    num_train = 0

    def __init__(self, num_train):
        self.num_train = num_train

    def add_results(self, loss):
        self.loss_history.append(loss)

    def set_model_state(self, model_state):
        self.model_state = model_state


class TrainArgs(object):
    """
    Structure to hold arguments used to be passed to training instances. Arguments should all be serializable in case
    the arguments are send over wire.
    """

    def __init__(self, epochs, *args, **kwargs):
        self.epochs = epochs
        # initialize without optimizer
        self.optimizer_class = None
        self.optimizer_args = None
        self.optimizer_kwargs = None
        # save additional argument to pass to training routine
        self.args = args
        self.kwargs = kwargs

    def set_optimizer(self, optimizer_class: Type[optim.Optimizer], *args, **kwargs):
        """
        Set an optimizer for this training
        :param optimizer_class:
        :param args:
        :param kwargs:
        :return:
        """
        self.optimizer_class = optimizer_class
        self.optimizer_args = args
        self.optimizer_kwargs = kwargs


class ModelArgs(object):

    def __init__(self, model_class: Type[nn.Module], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.model_class = model_class


class CriterionArgs(object):

    def __init__(self, criterion_class: Type[nn.Module], *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.criterion_class = criterion_class


class FederatedDatasetData(object):
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
