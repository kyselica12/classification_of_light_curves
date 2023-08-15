from pydoc import locate
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import glob
import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from abc import ABC, abstractmethod


from src.config import NetConfig, Config

class BaseNet(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.checkpoint = 0
        self.epoch_trained = 0
    
    @ abstractmethod
    def forward(self, x): 
        pass

    # def save(self):
    #     torch.save(self.state_dict(), f'{self.cfg.save_path}\\{self.name}_epochs_{self.epoch_trained:d}_checkpoint_{self.checkpoint:03d}.model')

    # def load(self, checkpoint='latest'):
    #     def get_checkpoint_and_epoch_number(path):
    #         checkpoint = int(os.path.split(path)[1][-9:-6])
    #         epoch = int(path.split("_")[-3])
    #         seed = int(path.split("_")[-5])
    #         return checkpoint, epoch, seed

    #     if checkpoint == 'latest':
    #         models = list(glob.iglob(f"{self.cfg.save_path}\\{self.name}_epochs_*_checkpoint_*.model"))
    #         model_path = max(models, key=get_checkpoint_and_epoch_number)
    #     else:
    #         models = list(glob.iglob(f"{self.cfg.save_path}\\{self.name}_epochs_*_checkpoint_{checkpoint:03d}.model"))
    #         model_path = max(models, key=get_checkpoint_and_epoch_number)

    #     self.load_state_dict(torch.load(model_path))
    #     self.checkpoint, self.epoch_trained, seed = get_checkpoint_and_epoch_number(model_path)

    #     return seed