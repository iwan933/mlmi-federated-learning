from typing import List, Dict

import torch

from mlmi.reptile.model import (
    ReptileClient, ReptileServer
)
from mlmi.exceptions import ClientError, ExecutionError

from mlmi.log import getLogger
from mlmi.struct import TrainArgs
from mlmi.utils import overwrite_participants_models
from mlmi.participant import BaseTrainingParticipant

logger = getLogger(__name__)

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
            #logger.debug('invoking training on participant {0}'.format(participant._name))
            participant.train(training_args)
            successful_participants += 1
        except Exception as e:
            logger.error('training on participant {0} failed'.format(participant._name), e)

    if success_threshold != -1 and successful_participants < success_threshold:
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')

def reptile_train_step(aggregator: ReptileServer,
                       participants: List[ReptileClient],
                       inner_training_args: TrainArgs,
                       meta_training_args: TrainArgs = None,
                       evaluation_mode: bool = False,
                       *args, **kwargs):
    """
    Routine to run a Reptile training step
    :param aggregator: aggregator participant that will aggregate the resulting training models
    :param participants: training participants in this round
    :param inner_training_args: training arguments for participant models
    :param meta_training_args: training arguments for meta model
    :return:
    """

    logger.debug('distribute the initial model to the clients.')
    initial_model_state = aggregator.model.state_dict()
    overwrite_participants_models(
        initial_model_state, participants, verbose=False
    )

    logger.debug('starting training round.')
    run_train_round(participants, inner_training_args)

    # Aggregate only when not in evaluation mode
    if not evaluation_mode:
        assert meta_training_args is not None, ('Argument meta_training_args '
            'must not be None when not in evaluation_mode')
        logger.debug('starting aggregation.')
        aggregator.aggregate(
            participants=participants,
            meta_learning_rate=meta_training_args.kwargs['meta_learning_rate']
        )
