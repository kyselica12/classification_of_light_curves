from copy import deepcopy

import numpy as np
import torch.nn as nn
import torch

from src.config import BasicCNNConfig, FCConfig, NetConfig
from src.nn.cnn import CNN
from src.nn.fc import FC
from src.nn.net import BaseNet

class CNNFC(BaseNet):

    def _initialize(self, cnn_args, fc_args):   
        cnn_config: BasicCNNConfig = BasicCNNConfig(**cnn_args)
        fc_config: FCConfig = FCConfig(**fc_args)

        #TODO: send good input size
        cnn_net_config: NetConfig = deepcopy(self.cfg)
        cnn_net_config.net_args = cnn_config.__dict__
        self.cnn = CNN(cnn_net_config)
        
        fc_net_config: NetConfig = deepcopy(self.cfg)
        fc_net_config.net_args = fc_config.__dict__
        self.fc = FC(fc_net_config)

        self.final_layer = nn.Linear(cnn_config.hid_dim + fc_config.layers[-1], self.n_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
   
    def forward(self, x):
        x = torch.reshape(x, (-1, self.size))
        
        x1 = self.cnn.layers(x) #TODO reshape to correct shape
        x2 = self.fc.layers(x)

        x = torch.cat((x1, x2), dim=1)
        self.final_layer(x)
        
        return self.logsoftmax(x)

