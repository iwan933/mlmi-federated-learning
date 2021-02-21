import threading
import json
from typing import Callable, List, Dict, Optional, Tuple, Union

import torch
from torch import Tensor

from mlmi.fedavg.structs import FedAvgExperimentContext
from mlmi.participant import (
    BaseTrainingParticipant, BaseAggregatorParticipant,
)
from mlmi.exceptions import ExecutionError, GradientExplodingError

from mlmi.log import getLogger
from mlmi.sampling import sample_randomly_by_fraction
from mlmi.settings import REPO_ROOT
from mlmi.structs import TrainArgs
from mlmi.utils import evaluate_global_model, overwrite_participants_models, overwrite_participants_optimizers

logger = getLogger(__name__)


def run_fedavg_train_round(
        initial_model_state: Dict[str, Tensor],
        participants: List['BaseTrainingParticipant'],
        training_args: TrainArgs,
        success_threshold=-1
) -> List['BaseTrainingParticipant']:
    """
    Routine to run a single round of training on the clients and return the results additional args are passed to the
    clients training routines.
    :param initial_model_state: model state to communicate before training
    :param participants: participants to train in this round
    :param training_args: arguments passed for training
    :param success_threshold: threshold for how many clients should at least participate in the round
    :return:
    """
    overwrite_participants_models(initial_model_state, participants)
    successful_participants = []
    for participant in participants:
        try:
            logger.debug(f'invoking training on participant {participant._name}')
            participant.train(training_args)
            successful_participants.append(participant)
            if success_threshold != -1 and success_threshold <= len(successful_participants):
                break
        except GradientExplodingError as gradient_exception:
            logger.error(f'participant {participant._name} failed due to exploding gradients', gradient_exception)
        except Exception as e:
            logger.error(f'training on participant {participant._name} failed', e)

    if success_threshold != -1 and len(successful_participants) < success_threshold:
        raise ExecutionError('Failed to execute training round, not enough clients participated successfully')
    return successful_participants


def run_fedavg_round(aggregator: 'BaseAggregatorParticipant', participants: List['BaseTrainingParticipant'],
                     training_args: TrainArgs, client_fraction=1.0):
    """
    Routine to run a training round with the given clients based on the server model and then aggregate the results
    :param client_fraction: client fraction to train with
    :param aggregator: aggregator participant that will aggregate the resulting training models
    :param participants: training participants in this round
    :param training_args: training arguments for this round
    :return:
    """
    logger.debug('distribute the initial model to the clients.')
    initial_model_state = aggregator.model.state_dict()

    success_threshold = max(int(len(participants) * client_fraction), 1) if client_fraction < 1.0 else -1
    participant_fraction = sample_randomly_by_fraction(participants, client_fraction)
    logger.debug(f'starting training round with {len(participant_fraction)}/{len(participants)}.')
    trained_participants = run_fedavg_train_round(initial_model_state, participant_fraction, training_args,
                                                  success_threshold=-1)

    logger.debug('starting aggregation.')
    num_train_samples = [p.num_train_samples for p in trained_participants]
    aggregator.aggregate(trained_participants, num_train_samples=num_train_samples)

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


def evaluate_cluster_models(cluster_server_dic: Dict[str, 'BaseAggregatorParticipant'],
                            cluster_clients_dic: Dict[str, List['BaseTrainingParticipant']])\
        -> Tuple[Tensor, Tensor]:
    global_losses = None
    global_acc = None
    for cluster_id, cluster_clients in cluster_clients_dic.items():
        cluster_server = cluster_server_dic[cluster_id]
        result = evaluate_global_model(global_model_participant=cluster_server, participants=cluster_clients)
        if global_losses is None:
            global_losses = result.get('test/loss')
            global_acc = result.get('test/acc')
        else:
            if global_losses.dim() == 0:
                global_losses = torch.tensor([global_losses])
            if global_acc.dim() == 0:
                global_acc = torch.tensor([global_acc])
            if result.get('test/loss').dim() == 0:
                loss_test = torch.tensor([result.get('test/loss')])
                global_losses = torch.cat((global_losses, loss_test), dim=0)
            else:
                global_losses = torch.cat((global_losses, result.get('test/loss')), dim=0)
            if result.get('test/acc').dim() == 0:
                acc_test = torch.tensor([result.get('test/acc')])
                global_acc = torch.cat((global_acc, acc_test), dim=0)
            else:
                global_acc = torch.cat((global_acc, result.get('test/acc')), dim=0)
    return global_losses, global_acc
