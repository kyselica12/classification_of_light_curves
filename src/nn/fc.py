import numpy as np
import torch.nn as nn
import torch

from src.nn.net import BaseNet


class FC(BaseNet):

    def _initialize(self, layers):   
        list_layers = []
        prev = self.size

        for i in range(len(layers)):
            list_layers.append(nn.Linear(prev, layers[i]))
            list_layers.append(nn.ReLU())
            prev = layers[i]

        self.final_layer = nn.Linear(prev, self.n_classes)
        self.layers = nn.Sequential(*list_layers)
        self.logsoftmax = nn.LogSoftmax(dim=1)
   
    def forward(self, x):
        x = torch.reshape(x, (-1, self.size))
    
        x = self.layers(x)
        x = self.final_layer(x)

        return self.logsoftmax(x)
