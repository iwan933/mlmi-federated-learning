import copy
import numpy as np
from typing import Dict, List, Optional

import torch
from torch import Tensor, optim, nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from fedml_api.model.cv.cnn import CNN_OriginalFedAvg

from mlmi.clustering import flatten_model_parameter
from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant
from mlmi.structs import OptimizerArgs


logger = getLogger(__name__)


def add_weighted_model(previous: Optional[Dict[str, Tensor]], next: Dict[str, Tensor], num_samples: int,
                       num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = dict() if previous is None else copy.deepcopy(previous)
    for key, w in next.items():
        weighted_parameter = (float(num_samples) / float(num_total_samples)) * w
        if previous is None:
            weighted_model_state[key] = weighted_parameter
        else:
            weighted_model_state[key] = weighted_model_state[key] + weighted_parameter
    return weighted_model_state


def load_participant_model_state(participant: BaseParticipant) -> Dict[str, Tensor]:
    """
    Method to load the current model state from a participant
    :param participant: participant to load the model from
    :return:
    """
    return participant.model.state_dict()


class FedAvgClient(BaseTrainingParticipant):
    pass


class FedAvgServer(BaseAggregatorParticipant):

    def aggregate(self, participants: List[BaseParticipant], num_train_samples: List[int] = None, *args, **kwargs):
        assert num_train_samples is not None, 'Place pass num_train_samples to the aggregation function.'
        assert len(num_train_samples) == len(participants), 'Please provide the keyword argument num_train_samples, ' \
                                                            'containing the number of training samples for each ' \
                                                            'participant'
        num_total_samples = sum(num_train_samples)

        aggregated_model_state = None
        keys = list(participants[0].model.state_dict().keys())
        sample_counter = 0
        for num_samples, participant in zip(num_train_samples, participants):
            sample_counter += num_samples
            assert num_samples == participant.num_train_samples, f'aggregation num: {num_samples} != ' \
                                                                       f'{participant.num_train_samples}'
            # flatten all dimensions and take the sum
            sum0 = np.sum(np.array([flatten_model_parameter(p.model.state_dict(), keys).numpy() for p in participants],
                                   dtype=float))

            aggregated_model_state = add_weighted_model(aggregated_model_state,
                                                        load_participant_model_state(participant),
                                                        num_samples, num_total_samples)

            sum1 = np.sum(np.array([flatten_model_parameter(p.model.state_dict(), keys).numpy() for p in participants],
                                   dtype=float))
            # check if the models were not influenced during aggregation
            assert sum0 == sum1, 'other models influenced'
        assert sample_counter == num_total_samples
        comparison = None
        for p in participants:
            if comparison is None:
                comparison = copy.deepcopy(p.model.state_dict())
            else:
                for k, v in p.model.state_dict().items():
                    comparison[k] += v
        self.overwrite_model_state(aggregated_model_state)


class CNNLightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, only_digits=False, *args, **kwargs):
        model = CNN_OriginalFedAvg(only_digits=only_digits)
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        self.accuracy = Accuracy()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # TODO: this should actually be calculated on a validation set (missing cross entropy implementation)
        self.log('train/acc/{}'.format(self.participant_name), self.accuracy(preds, y).item())
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f'test/acc/{self.participant_name}', self.accuracy(preds, y).item())
        self.log(f'test/loss/{self.participant_name}', loss.item())
        return loss


class CNNMnist(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x


class CNNMnistLightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, num_classes, *args, **kwargs):
        model = CNNMnist(input_channels=1, num_classes=num_classes)
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        self.accuracy = Accuracy()

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # TODO: this should actually be calculated on a validation set (missing cross entropy implementation)
        self.log('train/acc/{}'.format(self.participant_name), self.accuracy(preds, y).item())
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f'test/acc/{self.participant_name}', self.accuracy(preds, y).item())
        self.log(f'test/loss/{self.participant_name}', loss.item())
        return loss
