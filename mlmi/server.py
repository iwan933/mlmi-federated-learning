from typing import List, Dict
from torch import Tensor

from mlmi.struct import TrainArgs, TrainResults


class BaseServer(object):

    def aggregate(self, train_results: List[TrainResults], *args, **kwargs):
        raise NotImplementedError()

    def get_model_state(self) -> Dict[str, Tensor]:
        raise NotImplementedError()

    def get_training_arguments(self):
        raise NotImplementedError()


def load_server_model(server: BaseServer) -> Dict[str, Tensor]:
    """
    Load the server model
    :param server: server to load the model from
    :return: the servers model state
    """
    return server.get_model_state()


def load_server_training_arguments(server: BaseServer) -> TrainArgs:
    """
    Retrieves the default training argument from server.
    :param server: server to train for
    :return: training arguments specified for server
    """
    return server.get_training_arguments()
