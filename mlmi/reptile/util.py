from typing import List, Dict

import random
from torch import Tensor

from mlmi.reptile.model import (
    ReptileClient, ReptileServer
)
from mlmi.exceptions import ClientError, ExecutionError

from mlmi.log import getLogger
from mlmi.struct import TrainArgs
from mlmi.utils import overwrite_participants_models

logger = getLogger(__name__)


# TODO: meta_learning_rate
#       run_train_aggregate_round should require ReptileServer, clients in cluster,
#       learning_rate, meta_learning_rate, num_inner_iterations

def run_train_round(participants: List[BaseTrainingParticipant], training_args: TrainArgs, success_threshold=-1):
    """
    Routine to run a single round of training on the clients and return the results additional args are passed to the
    clients training routines.
    :param participants: participants to train in this round
    :param training_args: arguments passed for training
    :param success_threshold: threshold for how many clients should at least participate in the round
    :return:
    """
    successful_participants = 0
    for participant in participants:
        try:
            # invoke local training
            logger.debug('invoking training on participant {0}'.format(participant._name))
            participant.train(training_args)
            successful_participants += 1
        except Exception as e:
            logger.error('training on participant {0} failed'.format(participant._name), e)

    if success_threshold != -1 and successful_participants < success_threshold:
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')

def run_reptile_round(aggregator: ReptileServer, participants: List[ReptileClient],
                      training_args: TrainArgs, *args, **kwargs):
    """
    Routine to run a training round with the given clients based on the server model and then aggregate the results
    :param aggregator: aggregator participant that will aggregate the resulting training models
    :param participants: training participants in this round
    :param training_args: training arguments for this round
    :return:
    """

    # Sample subset of participants as meta batch
    assert isinstance(training_args.kwargs['meta_batch_size'], int), \
        'Error: meta_batch_size must be int'
    if training_args.kwargs['meta_batch_size'] == -1:
        meta_batch = participants
    else:
        meta_batch = random.sample(participants, training_args.kwargs['meta_batch_size'])

    logger.debug('distribute the initial model to the clients.')
    initial_model_state = aggregator.model.state_dict()
    overwrite_participants_models(initial_model_state, meta_batch)

    logger.debug('starting training round.')
    run_train_round(meta_batch, training_args)

    logger.debug('starting aggregation.')
    aggregator.aggregate(meta_batch, *args, **kwargs)

    logger.debug('distribute the aggregated global model to clients')
    resulting_model_state = aggregator.model.state_dict()
    overwrite_participants_models(resulting_model_state, participants)
