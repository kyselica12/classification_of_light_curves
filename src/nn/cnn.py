import numpy as np
import torch.nn as nn

from src.nn.net import BaseNet

class CNN(BaseNet):
    
    def _initialize(self, stride, kernel_size, n_channels, hid_dim):
        

        padding =  kernel_size // 2
        print("padding", padding)

        in_dim = int(np.floor((self.size + stride - 1) / stride))
        in_dim = int(np.floor((in_dim + 2 - 1)) / 2) * n_channels

        self.layers = nn.Sequential(
            nn.Conv1d(1, n_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            nn.Flatten(),
            # nn.Dropout(0.2),
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            # nn.Dropout(0.2),
        )

        self.final_layer = nn.Linear(hid_dim, self.n_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x): 
        out = x
        
        out = self.layers(out)
        out = self.final_layer(out)

        return self.logsoftmax(out)
