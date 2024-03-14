from typing import Any
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import wandb

import torch.nn as nn
import torch.optim as optim

from src.configs import NetConfig, NetArchitecture
from src.module.cnn import CNN
from src.module.cnnfc import CNNFC
from src.module.fc import FC


class LCModule(pl.LightningModule):

    def __init__ (self, cfg:NetConfig):
        super().__init__()
        self.cfg: NetConfig = cfg
        self.n_classes = cfg.output_size
        self.learning_rate = cfg.learning_rate
        self.net = self._initialize_net(cfg)
        self.net = self.net.float()

        self.log_confusion_matrix = False

        self.save_hyperparameters()
        self.criterion = nn.CrossEntropyLoss()

        self.val_preds = []
        self.val_target = []
        self.test_preds = []
        self.test_target = []
    
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

        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        _, preds, acc, loss = self._get_logit_pred_acc_loss(batch, batch_idx)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True)

        if self.log_confusion_matrix:
            self.val_preds.append(preds)
            self.val_target.append(batch[1])

        return loss

    def test_step(self, batch, batch_idx):
        _, preds, acc, loss = self._get_logit_pred_acc_loss(batch, batch_idx)

        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)

        if self.log_confusion_matrix:
            self.test_preds.append(preds)
            self.test_target.append(batch[1])

        return loss

    def _log_conf_matrix(self, preds, target, name=""):
        y_true = torch.concatenate(target).cpu().numpy()
        y_preds = torch.concatenate(preds).cpu().numpy()
        wandb.log({f'{name}_conf_mat': wandb.plot.confusion_matrix(
                            y_true=y_true,
                            preds=y_preds,
                            class_names=self.cfg.class_names)})   
        preds.clear()
        target.clear()
        

    def on_validation_epoch_end(self):
        if self.log_confusion_matrix:
            self._log_conf_matrix(self.val_preds, self.val_target, "val")
    
    def on_test_epoch_end(self) -> None:
        if self.log_confusion_matrix:
            self._log_conf_matrix(self.test_preds, self.test_target, "test")
    
    def _get_logit_pred_acc_loss(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        losses = self.criterion(logits, y.long())
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        return logits, preds, acc, losses

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


        