from torch import nn, optim
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter

from mlmi.struct import TrainResults, TrainArgs


class BaseTrainer(object):

    def train_model(self, model: nn.Module, dataloader: data.DataLoader, num_train_samples: int, training_args: TrainArgs,
                    optimizer: optim.Optimizer, criterion: nn.Module, tensorboard_writer: SummaryWriter) -> TrainResults:
        """
        Trains the model with the given data
        :param criterion: the criterion to calculate the loss
        :param num_train_samples: number of training samples in the dataset
        :param optimizer: optimizer instance to use
        :param tensorboard_writer: reference to tensorboard writer
        :param model: model to train
        :param dataloader: training dataloaer
        :param training_args: arguments for training
        :return:
        """
        raise NotImplementedError()
