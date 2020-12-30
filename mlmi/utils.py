from typing import Dict, List

import torch
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from mlmi.participant import BaseParticipant, BaseTrainingParticipant
from mlmi.settings import RUN_DIR
from mlmi.log import getLogger
from mlmi.struct import ExperimentContext

logger = getLogger(__name__)


def create_tensorboard_logger(context: 'ExperimentContext') -> TensorBoardLogger:
    """

    :param context: the experiment context to use to for name generation
    :return:
    """
    configuration_string = '{}_e{}bs{}lr{}cf{}'.format(context.dataset, context.local_epochs, context.batch_size,
                                                       context.lr, context.client_fraction)
    experiment_path = RUN_DIR / context.name / configuration_string
    return TensorBoardLogger(experiment_path.absolute())


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


def evaluate_local_models(participants: List['BaseTrainingParticipant']):
    test_losses = []
    test_acc = []
    for participant in participants:
        results = participant.test(use_local_model=True)
        for result in results:
            for key in result.keys():
                if key.startswith('test/loss'):
                    test_losses.append(result.get(key))
                elif key.startswith('test/acc'):
                    test_acc.append(result.get(key))
    losses = torch.squeeze(torch.FloatTensor(test_losses))
    acc = torch.squeeze(torch.FloatTensor(test_acc))
    return {'test/loss': losses, 'test/acc': acc}


def evaluate_global_model(global_model_participant: BaseParticipant, participants: List['BaseTrainingParticipant']):
    test_losses = []
    test_acc = []
    for participant in participants:
        results = participant.test(model=global_model_participant.model)
        for result in results:
            for key in result.keys():
                if key.startswith('test/loss'):
                    test_losses.append(result.get(key))
                elif key.startswith('test/acc'):
                    test_acc.append(result.get(key))
    losses = torch.squeeze(torch.FloatTensor(test_losses))
    acc = torch.squeeze(torch.FloatTensor(test_acc))
    return {'test/loss': losses, 'test/acc': acc}
