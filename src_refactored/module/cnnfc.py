import torch
from torch import nn

from src_refactored.configs import CNNFCConfig, FCConfig, CNNConfig
from src_refactored.module.cnn import CNN
from src_refactored.module.fc import FC
from src_refactored.module.net import BaseNet

class CNNFC(BaseNet):

    def __init__(self, cfg: CNNFCConfig):   
        super().__init__()
        self.cfg = cfg

        self.cnn = CNN(
            CNNConfig(
                input_size=cfg.cnn_input_size,
                output_size=cfg.output_size,
                in_channels=cfg.in_channels,
                conv_layers=cfg.cnn_layers,
                classifier_layers=[]
            )
        )
        self.fc = FC(
            FCConfig(
                input_size=cfg.input_size-cfg.cnn_input_size*cfg.in_channels,
                output_size=cfg.fc_output_dim,
                layers=cfg.fc_layers
            )
        )
        
        self.relu = nn.ReLU()
        
        self.classifier = FC(
            FCConfig(
                input_size=cfg.fc_output_dim + self.cnn.hid_dim,
                output_size=cfg.output_size,
                layers=cfg.classifier_layers,
            )
        )

   
    def forward(self, x, features=False):

        cnn_input = self.cfg.cnn_input_size * self.cfg.in_channels
        fc_in = x[:, :-cnn_input]
        cnn_in = x[:, -cnn_input:].reshape(-1, self.cfg.in_channels, self.cfg.cnn_input_size)
        
        fc_out = self.fc(fc_in)
        cnn_out = self.cnn.layers(cnn_in)

        x = torch.cat((fc_out, cnn_out), dim=1)
        
        x = self.relu(x)
        
        return self.classifier.forward(x, features)

