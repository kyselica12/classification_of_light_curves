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
        for out_ch, kernel in cfg.conv_layers:
            layers.extend([
                nn.Conv1d(in_ch, out_ch, kernel_size=kernel, stride=1, padding=kernel//2),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
            ])
            in_ch = out_ch

        self.hid_dim = (cfg.input_size // (2**len(cfg.conv_layers))) * cfg.conv_layers[-1][0]

        layers.append(nn.Flatten())
        self.layers = nn.Sequential(*layers)

        self.classifier = FC(FCConfig(input_size=self.hid_dim, output_size=cfg.output_size, layers=cfg.classifier_layers))

    
    # def _initialize(self, stride, kernel_size, n_channels, hid_dim):
        
    #     padding =  kernel_size // 2
    #     print("padding", padding)

    #     in_dim = int(np.floor((self.size + stride - 1) / stride))
    #     in_dim = int(np.floor((in_dim + 2 - 1)) / 2) * n_channels
            


    #     self.layers = nn.Sequential(
    #         nn.Conv1d(1, n_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    #         nn.ReLU(),
    #         nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
    #         nn.Flatten(),
    #         # nn.Dropout(0.2),
    #         nn.Linear(in_dim, hid_dim),
    #         nn.ReLU(),
    #         # nn.Dropout(0.2),
    #     )

    #     self.final_layer = nn.Linear(hid_dim, self.n_classes)


    def forward(self, x): 
        x = torch.reshape(x, (-1, self.cfg.in_channels, self.cfg.input_size))
        x = self.layers(x)
        x = self.classifier(x)
        return x
