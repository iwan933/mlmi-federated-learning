import pytorch_lightning as pl
import torchvision
import torch
from torch.nn import Linear, functional as F
from pytorch_lightning.metrics import Accuracy

from mlmi.participant import BaseParticipantModel


class ResNet18Lightning(BaseParticipantModel, pl.LightningModule):

    def __init__(self, num_classes, *args, **kwargs):
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = Linear(in_features=512, out_features=num_classes, bias=True)
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
