import torch.nn as nn
from abc import ABC, abstractmethod

class BaseNet(nn.Module, ABC):

    def __init__(self):
        super().__init__()
        self.checkpoint = 0
        self.epoch_trained = 0
    
    @ abstractmethod
    def forward(self, x): 
        pass
