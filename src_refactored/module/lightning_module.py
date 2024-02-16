from typing import Any
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch

import torch.nn as nn
import torch.optim as optim

from src_refactored.configs import NetConfig, NetArchitecture
from src_refactored.module.cnn import CNN
from src_refactored.module.cnnfc import CNNFC
from src_refactored.module.fc import FC


class LCModule(pl.LightningModule):

    def __init__ (self, cfg:NetConfig):
        super().__init__()
        self.cfg = cfg
        self.n_classes = cfg.output_size
        self.net = self._initialize_net(cfg)
        self.net = self.net.float()

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()
    
    def _initialize_net(self, cfg: NetConfig):
        match cfg.architecture:
            case NetArchitecture.FC:
                return FC(cfg.args)
            case NetArchitecture.CNN:
                return CNN(cfg.args)
            case NetArchitecture.CNNFC:
                return CNNFC(cfg.args)
            case _:
                raise ValueError(f"Unknown architecture {cfg.architecture}")

    def forward(self, x):
        logits = self.net(x.float())
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
        losses = self.criterion(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return logits, preds, acc, losses
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


        