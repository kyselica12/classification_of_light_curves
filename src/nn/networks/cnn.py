import numpy as np
import torch.nn as nn
import torch

from src.nn.networks.net import BaseNet
from src.nn.networks.fc import FC
from src.config import CNNConfig, FCConfig

class CNN(BaseNet):

    def __init__(self, cfg: CNNConfig):
        super().__init__()
        self.cfg = cfg
        layers = []

        in_ch = cfg.in_channels
        dim = cfg.input_size
        for out_ch, kernel, stride in cfg.conv_layers:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=kernel//2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            ])
            in_ch = out_ch
            dim = (dim + stride - 1) // stride
            dim = int(np.ceil(dim / 2))
            # print(dim, out_ch, dim * out_ch)

        self.hid_dim = dim * out_ch

        print("CNN middle dim:", self.hid_dim)

        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        self.classifier = FC(
            FCConfig(
                input_size=self.hid_dim, 
                output_size=cfg.output_size, 
                layers=cfg.classifier_layers,
            )
        )

    def forward(self, x): 
        # print("Input", x)
        x = torch.reshape(x, (-1, self.cfg.in_channels, self.cfg.input_size))
        x = self.layers(x)
        # print(x)
        x = self.classifier(x)
        # print(x)

        return x
