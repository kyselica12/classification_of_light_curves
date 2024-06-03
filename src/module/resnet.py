'''
1D version of ResNet from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

from src.configs import ResNetConfig

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet']

def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                    nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, padding_mode='circular'),
                    nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, in_channels=1):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.in_channels = in_channels

        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool1d(out, out.size()[2])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def get_embedding(self, x):
        x = x.reshape(x.shape[0], self.in_channels, -1)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        return out


def resnet20(num_classes=10, n_channels=1):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, n_channels)

def resnet32(num_classes=10, n_channels=1):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, n_channels)

def resnet44(num_classes=10, n_channels=1):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, n_channels)

def resnet(cfg: ResNetConfig):
    k = cfg.n_layers
    n = (k - 2) // 6
    return ResNet(BasicBlock, [n, n, n], cfg.output_size, cfg.in_channels)




