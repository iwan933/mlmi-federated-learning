from typing import Dict, List, Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

from mlmi.exceptions import GradientExplodingError
from mlmi.log import getLogger
from mlmi.participant import BaseParticipantModel, BaseTrainingParticipant, BaseAggregatorParticipant, BaseParticipant


logger = getLogger(__name__)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def add_weighted_model(previous: Optional[Dict[str, Tensor]], next: Dict[str, Tensor], num_samples: int,
                       num_total_samples: int) -> Dict[str, Tensor]:
    weighted_model_state = dict() if previous is None else previous
    for key, w in next.items():
        weighted_parameter = (num_samples / num_total_samples) * w
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

    def aggregate(self, participants: List['BaseTrainingParticipant'], *args, **kwargs):
        num_total_samples = sum([p.num_train_samples for p in participants])

        aggregated_model_state = None
        for participant in participants:
            aggregated_model_state = add_weighted_model(aggregated_model_state,
                                                        load_participant_model_state(participant),
                                                        participant.num_train_samples, num_total_samples)
        self.model.load_state_dict(aggregated_model_state)


class CNN_OriginalFedAvg(torch.nn.Module):

    def __init__(self, only_digits=True, input_channels=1):
        super(CNN_OriginalFedAvg, self).__init__()
        self.only_digits = only_digits
        self.conv2d_1 = torch.nn.Conv2d(input_channels, 32, kernel_size=5, padding=2)
        self.max_pooling = nn.MaxPool2d(2, stride=2)
        self.conv2d_2 = torch.nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.flatten = nn.Flatten()
        self.linear_1 = nn.Linear(3136, 512)
        self.linear_2 = nn.Linear(512, 10 if only_digits else 62)

    def forward(self, x):
        x = self.conv2d_1(x)
        x = F.relu(self.max_pooling(x))
        x = self.conv2d_2(x)
        x = F.relu(self.max_pooling(x))
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.linear_2(x)
        return x


class CNNLightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, only_digits=False, input_channels=1, *args, **kwargs):
        model = CNN_OriginalFedAvg(only_digits=only_digits, input_channels=input_channels)
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        # self.model.apply(init_weights)
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
        if torch.isnan(loss) or torch.isinf(loss):
            raise GradientExplodingError('Loss is nan or inf, it seems gradient exploded.')
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f'test/acc/{self.participant_name}', self.accuracy(preds, y).item())
        self.log(f'test/loss/{self.participant_name}', loss.item())


class CNNMnist(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = self.flatten(x)
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
        self.log('train/acc/{}'.format(self.participant_name), self.accuracy(preds, y))
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log(f'test/acc/{self.participant_name}', self.accuracy(preds, y))
        self.log(f'test/loss/{self.participant_name}', loss.item())
        return loss
