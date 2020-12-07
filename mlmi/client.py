from typing import Type, Dict

from torch import nn, optim, Tensor
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from mlmi.model.trainer import BaseTrainer
from mlmi.struct import TrainResults, TrainArgs, ModelArgs, CriterionArgs
from mlmi.utils import create_tensorboard_writer


class BaseClient(object):
    def __init__(self, client_id: str, train_dataloader: data.DataLoader, num_train_samples: int,
                 test_dataloader: data.DataLoader, num_test_samples: int, model_args: ModelArgs,
                 criterion_args: CriterionArgs, *args, **kwargs):
        self.client_id = client_id
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        self.model = model_args.model_class(*model_args.args, **model_args.kwargs)
        self.criterion = criterion_args.criterion_class(*criterion_args.args, **criterion_args.kwargs)
        self.tensorboard_writer = create_tensorboard_writer('client' + self.get_id())
        self.args = args
        self.kwargs = kwargs
        self.training = False
        self.train_results = None
        self.model_state = None

    def get_id(self) -> str:
        return self.client_id

    def get_trainer(self) -> BaseTrainer:
        raise NotImplementedError()

    def get_model(self) -> nn.Module:
        return self.model

    def get_optimizer(self) -> optim.Optimizer:
        raise NotImplementedError()

    def get_tensorboard_writer(self) -> SummaryWriter:
        raise NotImplementedError()

    def get_train_results(self) -> TrainResults:
        return self.train_results

    def load_model_state(self, model_state: Dict[str, Tensor]):
        self.model_state = model_state

    def get_train_data_loader(self) -> data.DataLoader:
        return self.train_dataloader

    def get_test_data_loader(self) -> data.DataLoader:
        return self.test_dataloader

    def get_num_train_samples(self) -> int:
        return self.num_train_samples

    def get_num_test_samples(self) -> int:
        return self.num_test_samples

    def get_criterion(self):
        return self.criterion

    def train(self, training_args, *args, **kwargs):
        try:
            assert self.model_state is not None, 'Model state needs to be loaded before training'
            self.training = True
            self.train_results = None
            trainer = self.get_trainer()
            local_model = self.get_model()
            local_model.load_state_dict(self.model_state)
            optimizer = self.get_optimizer()
            train_dataloader = self.get_train_data_loader()
            if optimizer is None:
                optimizer_class = training_args.optimizer_class
                optimizer = optimizer_class(local_model.parameters(), *training_args.optimizer_args,
                                            **training_args.optimizer_kwargs)
            local_model.train()
            self.train_results = trainer.train_model(local_model, dataloader=train_dataloader, optimizer=optimizer,
                                                     training_args=training_args, criterion=self.get_criterion(),
                                                     num_train_samples=self.get_num_train_samples(),
                                                     tensorboard_writer=self.get_tensorboard_writer(),
                                                     *args, **kwargs)
        except Exception as e:
            raise e
        finally:
            self.training = False

    def save_training_results(self):
        # TODO save training results persistently
        raise NotImplementedError()

    def load_training_results(self):
        # TODO load training results from persistent storage
        raise NotImplementedError()


def create_client(client_class: Type[BaseClient], client_id: str, *args, **kwargs) -> BaseClient:
    """
    :param client_class:
    :param client_id:
    :param args:
    :param kwargs:
    :return:
    """
    return client_class(client_id, *args, **kwargs)


def send_model_to_client(client: BaseClient, model_state: Dict[str, Tensor]):
    """
    Sends model to client
    :param client: client to send the model to
    :param model_state: state of the client model
    :return:
    """
    client.load_model_state(model_state)


def retrieve_train_results(client: BaseClient) -> TrainResults:
    """
    Retrieve the model state from the client (pull structure)
    :param client: client to pull the model state from
    :return: the clients model state
    """
    return client.get_train_results()


def invoke_client_training(client: BaseClient, training_args: TrainArgs):
    """
    Invokes training on the client and passes additional arguments that can be used for training
    :param training_args: arguments passed to clients training routine
    :param client: client to run the training on
    :return:
    """
    client.train(training_args)
