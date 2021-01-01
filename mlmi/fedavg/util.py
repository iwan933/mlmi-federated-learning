from typing import List, Dict, Union

import torch
from torch import Tensor

from mlmi.participant import (
    BaseTrainingParticipant, BaseAggregatorParticipant,
)
from mlmi.exceptions import ExecutionError

from mlmi.log import getLogger
from mlmi.settings import REPO_ROOT
from mlmi.struct import TrainArgs, ExperimentContext
from mlmi.utils import overwrite_participants_models

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
            logger.debug('invoking training on participant {0}'.format(participant._name))
            participant.train(training_args)
            successful_participants += 1
        except Exception as e:
            logger.error('training on participant {0} failed'.format(participant._name), e)

    if success_threshold != -1 and successful_participants < success_threshold:
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')


def run_fedavg_round(aggregator: BaseAggregatorParticipant, participants: List[BaseTrainingParticipant],
                     training_args: TrainArgs, *args, **kwargs):
    """
    Routine to run a training round with the given clients based on the server model and then aggregate the results
    :param aggregator: aggregator participant that will aggregate the resulting training models
    :param participants: training participants in this round
    :param training_args: training arguments for this round
    :return:
    """
    logger.debug('distribute the initial model to the clients.')
    initial_model_state = aggregator.model.state_dict()
    overwrite_participants_models(initial_model_state, participants)

    logger.debug('starting training round.')
    run_train_round(participants, training_args)

    logger.debug('starting aggregation.')
    aggregator.aggregate(participants, *args, **kwargs)

    logger.debug('distribute the aggregated global model to clients')
    resulting_model_state = aggregator.model.state_dict()
    overwrite_participants_models(resulting_model_state, participants)


def save_fedavg_state(experiment_context: 'ExperimentContext', fl_round: int, model_state: Dict[str, Tensor]):
    dataset = experiment_context.dataset
    client_fraction = experiment_context.client_fraction
    local_epochs = experiment_context.local_epochs
    lr = experiment_context.lr
    batch_size = experiment_context.batch_size
    path = REPO_ROOT / 'run' / 'states' / 'fedavg' / f'{dataset.name}_bs{batch_size}lr{lr:.2E}cf{client_fraction:.2f}e{local_epochs}r{fl_round}.mdl'
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)


def load_fedavg_state(experiment_context: 'ExperimentContext', fl_round: int) -> Union[Dict[str, Tensor], None]:
    dataset = experiment_context.dataset
    client_fraction = experiment_context.client_fraction
    local_epochs = experiment_context.local_epochs
    lr = experiment_context.lr
    batch_size = experiment_context.batch_size
    path = REPO_ROOT / 'run' / 'states' / 'fedavg' / f'{dataset.name}_bs{batch_size}lr{lr:.2E}cf{client_fraction:.2f}e{local_epochs}r{fl_round}.mdl'
    if not path.exists():
        return None
    return torch.load(path)
