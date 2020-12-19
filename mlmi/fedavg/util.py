from typing import List, Dict

from torch import Tensor

from mlmi.participant import (
    BaseTrainingParticipant, BaseAggregatorParticipant,
)
from mlmi.exceptions import ClientError, ExecutionError

from mlmi.log import getLogger
from mlmi.struct import TrainArgs


logger = getLogger(__name__)


def run_train_round(initial_model_state: Dict[str, Tensor], participants: List[BaseTrainingParticipant],
                    training_args: TrainArgs, success_threshold=-1):
    """
    Routine to run a single round of training on the clients and return the results additional args are passed to the
    clients training routines.
    :param initial_model_state: initial state of the model that should be applied to the clients
    :param participants: participants to train in this round
    :param training_args: arguments passed for training
    :param success_threshold: threshold for how many clients should at least participate in the round
    :return:
    """
    # distribute initial model state to clients
    successful_participants = 0
    for participant in participants:
        try:
            logger.debug('sending model to participant {0}'.format(participant._name))
            participant.load_model_state(initial_model_state)
            # invoke local training
            logger.debug('invoking training on participant {0}'.format(participant._name))
            participant.train(training_args)
            successful_participants += 1
        except Exception as e:
            logger.error('training on participant {0} failed'.format(participant._name), e)

    if success_threshold != -1 and successful_participants < success_threshold:
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')


def run_train_aggregate_round(aggregator: BaseAggregatorParticipant, participants: List[BaseTrainingParticipant],
                              training_args: TrainArgs, *args, **kwargs):
    """
    Routine to run a training round with the given clients based on the server model and then aggregate the results
    :param aggregator: aggregator participant that will aggregate the resulting training models
    :param participants: training participants in this round
    :param training_args: training arguments for this round
    :return:
    """
    initial_model_state = aggregator.get_model().state_dict()

    logger.debug('starting training round.')
    run_train_round(initial_model_state, participants, training_args)

    logger.debug('starting aggregation.')
    aggregator.aggregate(participants, *args, **kwargs)
