from typing import Dict, List, Optional

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from mlmi.participant import BaseParticipant, BaseTrainingParticipant
from mlmi.settings import RUN_DIR
from mlmi.log import getLogger


logger = getLogger(__name__)


def create_tensorboard_logger(experiment_name: str, experiment_specification: Optional[str] = None,
                              version=None) -> TensorBoardLogger:
    """

    :param experiment_name: name used for experiment
    :param experiment_specification: specification for experiment configuration
    :param version: allows to fix a version
    :return:
    """

    experiment_path = RUN_DIR / experiment_name
    if experiment_specification:
        experiment_path /= experiment_specification
    return TensorBoardLogger(experiment_path.absolute(), version=version)


def overwrite_participants_models(model_state: Dict[str, Tensor], participants):
    """
    Overwrites the participants models with a initial state
    :param model_state: state to save on the client side
    :param participants: list of participants to apply the model to
    :return:
    """
    for participant in participants:
        try:
            logger.debug('sending model to participant {0}'.format(participant._name))
            participant.overwrite_model_state(model_state)
        except Exception as e:
            logger.error('sendign model to participant {0} failed'.format(participant._name), e)


def _evaluate_model(participants: List['BaseTrainingParticipant'], test_on_participant):
    test_losses = []
    test_acc = []
    num_participants = len(participants)
    logger.debug('testing model ...')
    for i, participant in enumerate(participants):
        results = test_on_participant(participant)
        logger.debug(f'... tested model on {i+1:<4}/{num_participants} participants')
        for result in results:
            for key in result.keys():
                if key.startswith('test/loss'):
                    test_losses.append(result.get(key))
                elif key.startswith('test/acc'):
                    test_acc.append(result.get(key))
    losses = torch.squeeze(torch.FloatTensor(test_losses))
    acc = torch.squeeze(torch.FloatTensor(test_acc))
    return losses, acc


def evaluate_local_models(participants: List['BaseTrainingParticipant']):

    def _eval(participant):
        return participant.test(use_local_model=True)

    losses, acc = _evaluate_model(participants, _eval)
    return {'test/loss': losses, 'test/acc': acc}


def evaluate_global_model(global_model_participant: BaseParticipant, participants: List['BaseTrainingParticipant']):

    def _eval(participant):
        return participant.test(model=global_model_participant.model)

    losses, acc = _evaluate_model(participants, _eval)
    return {'test/loss': losses, 'test/acc': acc}


def fix_random_seeds(seed: int):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
