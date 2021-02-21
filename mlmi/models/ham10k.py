from typing import Any, List

import pytorch_lightning as pl
import torchvision
import torch
from torch.nn import CrossEntropyLoss, Dropout, Linear, Sequential, functional as F
from pytorch_lightning.metrics import Accuracy

from mlmi.participant import BaseParticipantModel


class ResNet18Lightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, num_classes, *args, weights=None, **kwargs):
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = Linear(in_features=512, out_features=num_classes, bias=True)
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        self.accuracy = Accuracy()
        self.train_accuracy = Accuracy()
        self.criterion = CrossEntropyLoss(weight=weights)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('train/acc/{}'.format(self.participant_name), self.train_accuracy(preds, y))
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, y)
        return {'loss': loss}

    def test_epoch_end(
            self, outputs: List[Any]
    ) -> None:
        loss_list = [o['loss'] for o in outputs]
        loss = torch.stack(loss_list)
        self.log(f'sample_num', self.accuracy.total.item())
        self.log(f'test/acc/{self.participant_name}', self.accuracy.compute())
        self.log(f'test/loss/{self.participant_name}', loss.mean().item())


class Densenet121Lightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, num_classes, *args, weights=None, **kwargs):
        model = torchvision.models.densenet121(pretrained=True)
        model.classifier = Linear(in_features=1024, out_features=num_classes, bias=True)
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        self.accuracy = Accuracy()
        self.train_accuracy = Accuracy()
        self.criterion = CrossEntropyLoss(weight=weights)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('train/acc/{}'.format(self.participant_name), self.train_accuracy(preds, y))
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, y)
        return {'loss': loss}

    def test_epoch_end(
            self, outputs: List[Any]
    ) -> None:
        loss_list = [o['loss'] for o in outputs]
        loss = torch.stack(loss_list)
        self.log(f'sample_num', self.accuracy.total.item())
        self.log(f'test/acc/{self.participant_name}', self.accuracy.compute())
        self.log(f'test/loss/{self.participant_name}', loss.mean().item())


class MobileNetV2Lightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, num_classes, *args, weights=None, **kwargs):
        model = torchvision.models.mobilenet_v2(pretrained=True)
        model.classifier = Sequential(
            Dropout(p=0.2, inplace=False),
            Linear(in_features=1280, out_features=num_classes, bias=True)
        )
        super().__init__(*args, model=model, **kwargs)
        self.model = model
        self.accuracy = Accuracy()
        self.train_accuracy = Accuracy()
        self.criterion = CrossEntropyLoss(weight=weights)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.log('train/acc/{}'.format(self.participant_name), self.train_accuracy(preds, y))
        self.log('train/loss/{}'.format(self.participant_name), loss.item())
        return loss

    def test_step(self, test_batch, batch_idx):
        x, y = test_batch
        y = y.long()
        logits = self.model(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy.update(preds, y)
        return {'loss': loss}

    def test_epoch_end(
            self, outputs: List[Any]
    ) -> None:
        loss_list = [o['loss'] for o in outputs]
        loss = torch.stack(loss_list)
        self.log(f'sample_num', self.accuracy.total.item())
        self.log(f'test/acc/{self.participant_name}', self.accuracy.compute())
        self.log(f'test/loss/{self.participant_name}', loss.mean().item())
