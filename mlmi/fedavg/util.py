import threading
import json
from typing import List, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.participant import (
    BaseTrainingParticipant, BaseAggregatorParticipant,
)
from mlmi.exceptions import ExecutionError

from mlmi.log import getLogger
from mlmi.settings import REPO_ROOT
from mlmi.structs import TrainArgs
from mlmi.utils import overwrite_participants_models, overwrite_participants_optimizers

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
    # reset gradients // first check how to handle optimizers in federated learning
    # aggregator.model.optimizer.zero_grad()
    # optimizer_state = aggregator.model.optimizer.state_dict()
    # overwrite_participants_optimizers(optimizer_state, participants)

    logger.debug('starting training round.')
    run_train_round(participants, training_args)

    logger.debug('starting aggregation.')
    aggregator.aggregate(participants, *args, **kwargs)

    logger.debug('distribute the aggregated global model to clients')
    resulting_model_state = aggregator.model.state_dict()
    overwrite_participants_models(resulting_model_state, participants)


def save_fedavg_hierarchical_cluster_configuration(
        experiment_context: FedAvgExperimentContext,
        cluster_names: List[str],
        client_ids: Dict[str, List[str]]
):
    # save client assignments
    configuration = {
        'cluster_names': cluster_names,
        'client_ids': client_ids
    }
    directory = REPO_ROOT / 'run' / 'states' / 'fedavg_hierarchical'
    configuration_path = directory / f'{experiment_context}{experiment_context.cluster_args}.json'
    configuration_path.parent.mkdir(parents=True, exist_ok=True)
    with open(configuration_path, 'w') as f:
        json.dump(configuration, f)


def load_fedavg_hierarchical_cluster_configuration(
        experiment_context: FedAvgExperimentContext
) -> Tuple[Optional[List[str]], Optional[Dict[str, List[str]]]]:
    # save client assignments

    directory = REPO_ROOT / 'run' / 'states' / 'fedavg_hierarchical'
    configuration_path = directory / f'{experiment_context}{experiment_context.cluster_args}.json'
    if not configuration_path.exists():
        return None, None
    with open(configuration_path, 'r') as f:
        configuration = json.load(f)
    return configuration.get('cluster_names'), configuration.get('client_ids')


def save_fedavg_hierarchical_cluster_model_state(
        experiment_context: FedAvgExperimentContext,
        fl_round: int,
        cluster_name: str,
        cluster_round: int,
        cluster_model_state: Dict[str, Tensor]
):
    """
    Saves the federated hierarchical clustering state (client distribution, cluster state)
    :param experiment_context: federated averaging experiment context
    :param fl_round: round of initializing fedavg
    :param cluster_model_state: state of the cluster model
    :param cluster_round: round inside cluster
    :param cluster_name: name of cluster
    :return:
    """

    unique_name = f'{experiment_context}r{fl_round}_{experiment_context.cluster_args}r{cluster_round}n{cluster_name}'
    directory = REPO_ROOT / 'run' / 'states' / 'fedavg_hierarchical'
    model_path = directory / f'{unique_name}.mdl'
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cluster_model_state, model_path)


def load_fedavg_hierarchical_cluster_model_state(
        experiment_context: FedAvgExperimentContext,
        fl_round: int, cluster_name: str,
        cluster_round: int
) -> Optional[Dict[str, Tensor]]:
    unique_name = f'{experiment_context}r{fl_round}_{experiment_context.cluster_args}r{cluster_round}n{cluster_name}'
    directory = REPO_ROOT / 'run' / 'states' / 'fedavg_hierarchical'
    model_path = directory / f'{unique_name}.mdl'
    if not model_path.exists():
        return None
    return torch.load(model_path)


def load_fedavg_hierarchical_cluster_state(
        experiment_context: 'FedAvgExperimentContext', fl_round: int) -> Union[Dict[str, Tensor], None]:
    path = REPO_ROOT / 'run' / 'states' / 'fedavg' / f'{experiment_context}r{fl_round}.mdl'
    if not path.exists():
        return None
    return torch.load(path)


def load_fedavg_state(experiment_context: 'FedAvgExperimentContext', fl_round: int) -> Union[Dict[str, Tensor], None]:
    path = REPO_ROOT / 'run' / 'states' / 'fedavg' / f'{experiment_context}r{fl_round}.mdl'
    if not path.exists():
        return None
    return torch.load(path)


def save_fedavg_state(experiment_context: 'FedAvgExperimentContext', fl_round: int, model_state: Dict[str, Tensor]):
    path = REPO_ROOT / 'run' / 'states' / 'fedavg' / f'{experiment_context}r{fl_round}.mdl'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, path)
