from typing import Any
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

import torch.nn as nn
import torch.optim as optim


class LCModule(pl.LightningModule):

    def __init__ (self, n_classes, net):
        self.net = net
        self.n_classes = n_classes

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logits = self.net(x)
        return logits

    def training_step(self, batch, batch_idx):
        _, _, acc, loss = self._get_logit_pred_acc_loss(batch, batch_idx)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _,_,acc, loss = self._get_logit_pred_acc_loss(batch, batch_idx)

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    
    def _get_logit_pred_acc_loss(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        losses = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return logits, preds, acc, losses


        