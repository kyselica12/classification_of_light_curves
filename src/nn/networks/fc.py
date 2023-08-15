import numpy as np
import torch.nn as nn
import torch

from src.nn.networks.net import BaseNet
from src.config import FCConfig

class FC(BaseNet):

    def __init__(self, cfg: FCConfig):
        super().__init__()
        self.cfg = cfg
        layers_config = [cfg.input_size] + cfg.layers + [cfg.output_size]

        layers = []

        for i in range(1, len(layers_config)):
            layers.append(nn.Linear(layers_config[i-1], layers_config[i]))
            layers.append(nn.ReLU())

        self.layers = nn.Sequential(*layers)

   
    def forward(self, x):
        x = torch.reshape(x, (-1, self.cfg.input_size))    
        x = self.layers(x)
        return x
