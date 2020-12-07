from typing import List, Dict

from torch import Tensor

from mlmi.client import (
    BaseClient,
    send_model_to_client,
    invoke_client_training,
    retrieve_train_results,
)
from mlmi.exceptions import ClientError, ExecutionError

from mlmi.log import getLogger
from mlmi.server import BaseServer, load_server_model, load_server_training_arguments
from mlmi.struct import TrainResults, TrainArgs

logger = getLogger(__name__)


def run_train_round(initial_model_state: Dict[str, Tensor], clients: List[BaseClient], training_args: TrainArgs,
                    success_threshold=-1) \
        -> List[TrainResults]:
    """
    Routine to run a single round of training on the clients and return the results additional args are passed to the
    clients training routines.
    :param initial_model_state: initial state of the model that should be applied to the clients
    :param clients: clients to use for this round
    :param training_args: arguments passed for training
    :param success_threshold: threshold for how many clients should at least participate in the round
    :return:
    """
    # distribute initial model state to clients
    for client in clients:
        logger.debug('sending model to client {0}'.format(client.get_id()))
        send_model_to_client(client, initial_model_state)
        # invoke local training
        logger.debug('invoking training on client {0}'.format(client.get_id()))
        invoke_client_training(client, training_args)

    # collect results
    train_results_list = []
    for client in clients:
        try:
            logger.debug('retrieving training results from client {0}'.format(client.get_id()))
            train_results = retrieve_train_results(client)
            train_results_list.append(train_results)
        except ClientError as e:
            logger.error('Client with id {0} failed'.format(client.get_id()), e)
    if success_threshold != -1 and len(train_results_list) < success_threshold:
        # TODO: remove last training results from clients
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')
    return train_results_list


def run_aggregation_round(server: BaseServer, clients: List[BaseClient]):
    # collect results
    train_results_list = []
    for client in clients:
        try:
            logger.debug('retrieving training results from client {0}'.format(client.get_id()))
            train_results = retrieve_train_results(client)
            train_results_list.append(train_results)
        except ClientError as e:
            logger.error('Client with id {0} failed'.format(client.get_id()), e)


def run_train_aggregate_round(server: BaseServer, clients: List[BaseClient]):
    """
    Routine to run a training round with the given clients based on the server model and then aggregate the results
    :param server: server that the round is run with
    :param clients: clients that should participate in the round
    :return:
    """
    initial_model_state = load_server_model(server)
    training_args = load_server_training_arguments(server)

    assert training_args
    assert initial_model_state

    logger.debug('starting training round.')
    training_result_list = run_train_round(initial_model_state, clients, training_args)

    logger.debug('starting aggregation.')
    server.aggregate(training_result_list)
