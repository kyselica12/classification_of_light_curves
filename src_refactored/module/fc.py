import numpy as np
import torch.nn as nn
import torch

from src_refactored.module.net import BaseNet
from src_refactored.configs import FCConfig

class FC(BaseNet):

    def __init__(self, cfg: FCConfig):
        super().__init__()
        self.cfg = cfg
        layers_config = [cfg.input_size] + cfg.layers + [cfg.output_size]

        layers = []
        
        for i in range(1, len(layers_config)):
            layer = nn.Linear(layers_config[i-1], layers_config[i])
            gain = nn.init.calculate_gain('relu')
            layers.append(layer)
            
            if i < len(layers_config) - 1:
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                layers.append(nn.ReLU())

        self.layers = nn.ModuleList(layers)

    def forward(self, x, features=False):
        x = torch.reshape(x, (-1, self.cfg.input_size))    
        for layer in self.layers[:-1]:
            x = layer(x)
        out = self.layers[-1](x)

        if features:
            return x, out

        return out
