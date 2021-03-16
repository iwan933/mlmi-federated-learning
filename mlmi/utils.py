import random
from typing import Dict, List, Optional

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from mlmi.models.ham10k import GlobalConfusionMatrix
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
    :param verbose: if true logs model sending
    :return:
    """
    for participant in participants:
        try:
            participant.overwrite_model_state(model_state)
        except Exception as e:
            logger.error('sending model to participant {0} failed'.format(participant._name), e)


def overwrite_participants_optimizers(optimizer_state: Dict[str, Tensor], participants):
    """
    Overwrites the participants optimizer with a state
    :param optimizer_state: state to save on the client side
    :param participants: list of participants to apply the model to
    :return:
    """
    for participant in participants:
        try:
            participant.overwrite_optimizer_state(optimizer_state)
        except Exception as e:
            logger.error('sendign model to participant {0} failed'.format(participant._name), e)


def _evaluate_model(participants: List['BaseTrainingParticipant'], model):
    test_losses = []
    test_acc = []
    test_balanced_acc = []
    num_participants = len(participants)
    num_samples_list = []
    logger.debug('testing model ...')
    for i, participant in enumerate(participants):
        if model is None:
            results = participant.test(use_local_model=True)
        else:
            results = participant.test(model)
        logger.debug(f'... tested model on {i+1:<4}/{num_participants} participants')
        for result in results:
            num_samples = result.get('sample_num') or 1
            num_samples_list.append(num_samples)
            for key in result.keys():
                if key.startswith('test/loss'):
                    test_losses.append(result.get(key))
                elif key.startswith('test/acc'):
                    test_acc.append(result.get(key))
                elif key.startswith('test/balanced_acc'):
                    test_balanced_acc.append(result.get(key))
    num_samples_total = sum(num_samples_list)
    losses = torch.squeeze(torch.FloatTensor(test_losses)).cpu()
    acc = torch.squeeze(torch.FloatTensor(test_acc)).cpu()
    balanced_acc = torch.squeeze(torch.FloatTensor(test_balanced_acc)).cpu()
    return losses, acc, balanced_acc, num_samples_total


def evaluate_local_models(participants: List['BaseTrainingParticipant']):
    global_confusion_matrix = GlobalConfusionMatrix()
    global_confusion_matrix.enable_logging()
    losses, acc, balanced_acc, num_samples = _evaluate_model(participants, None)
    global_confusion_matrix.disable_logging()
    return {
        'test/loss': losses,
        'test/acc': acc,
        'test/balanced_acc': balanced_acc,
        'num_samples': num_samples
    }


def evaluate_global_model(global_model_participant: 'BaseParticipant', participants: List['BaseTrainingParticipant']):
    global_confusion_matrix = GlobalConfusionMatrix()
    global_confusion_matrix.enable_logging()
    losses, acc, balanced_acc, num_samples = _evaluate_model(participants, global_model_participant.model)
    global_confusion_matrix.disable_logging()
    return {
        'test/loss': losses,
        'test/acc': acc,
        'test/balanced_acc': balanced_acc,
        'num_samples': num_samples
    }


def fix_random_seeds(seed: int):
    import numpy as np
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
