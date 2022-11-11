from torch import optim
import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
from torch.utils.data import DataLoader
import glob
import os

from config import NetConfig

import numpy as np
from sklearn.metrics import confusion_matrix


class Net(nn.Module):
    
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.cfg = cfg
        self._initialize()
        self.name = cfg.name

        if self.cfg.checkpoint:
            self.load(checkpoint=self.cfg.checkpoint)

        self.double()
        self.device = self.cfg.device
        self.to(self.device)
        

    def _initialize(self):
        self.checkpoint = 0
        
        self.size = self.cfg.input_size

        padding =  self.cfg.kernel_size // 2
        in_dim = int((self.size + padding* 2 - (self.cfg.kernel_size//2)*2) / self.cfg.stride / 4) * self.cfg.n_channels
        print("middle_dim", in_dim)

        self.layers = nn.Sequential(
            nn.Conv1d(1, self.cfg.n_channels, kernel_size=self.cfg.kernel_size, stride=self.cfg.stride, padding=padding),
            nn.MaxPool1d(kernel_size=5, stride=4, padding=1),
            nn.Flatten(),
            nn.Linear(in_dim, self.cfg.hid_dim),
            nn.ReLU(),
            nn.Linear(self.cfg.hid_dim, self.cfg.n_classes)
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)


        self.optim = None
        self.epoch_trained = 0

    def forward(self, x): 
        out = x
        for layer in self.layers:
            out = layer(out)
        
        return self.logsoftmax(out)

    def save(self):
        torch.save(self.state_dict(), f'{self.cfg.save_path}\\{self.name}_epochs_{self.epoch_trained:d}_checkpoint_{self.checkpoint:03d}.model')

    def load(self, checkpoint='latest', ):
        def get_checkpoint_and_epoch_number(path):
            checkpoint = int(os.path.split(path)[1][-9:-6])
            epoch = int(path.split("_")[-3])
            seed = int(path.split("_")[-5])
            return checkpoint, epoch, seed

        if checkpoint == 'latest':
            models = list(glob.iglob(f"{self.cfg.save_path}\\{self.name}_epochs_*_checkpoint_*.model"))
            model_path = max(models, key=get_checkpoint_and_epoch_number)
        else:
            models = list(glob.iglob(f"{self.cfg.save_path}\\{self.name}_epochs_*_checkpoint_{checkpoint:03d}.model"))
            model_path = max(models, key=get_checkpoint_and_epoch_number)

        self.load_state_dict(torch.load(model_path))
        self.checkpoint, self.epoch_trained, seed = get_checkpoint_and_epoch_number(model_path)

        return seed

    def _set_optim(self, fresh=False):
        if self.optim is None or fresh:
            self.optim = optim.Adam(self.parameters(), lr=0.001)

    def train_model(self, 
              train_loader, val_loader, epochs, 
              reset_optim=False,
              tensorboard=False,
              save_interval=None, 
              show=False,
              class_weights=None):

        self.train()
        self._set_optim(reset_optim)

        criterion = nn.NLLLoss(weight=class_weights) 

        if tensorboard:
            tensorboard = SummaryWriter(log_dir="C:/Users/Kyselica/Desktop/kyselica/classification_of_light_curves/tensorboard/run")

        start_checkpoint = self.checkpoint

        for epoch in tqdm.tqdm(range(epochs), desc="Training", position=0):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0

            for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}: ", leave=False, position=1):
                # get the inputs; data is a list of [inputs, labels]
                epoch_loss, epoch_correct = self._train_one_epoch(criterion, data)

                # print statistics
                running_loss += epoch_loss
                correct += epoch_correct
                total += data[1].size(0)

            running_loss /= total

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save()
                self.checkpoint += 1

            val_acc, val_loss, _ = evaulate_net(self, val_loader)

            if show:
                print(f"Train:\n\tLoss: {running_loss}\n\tAcc: {correct/total*100}", flush=True)
                print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)

            self.epoch_trained += 1

            if tensorboard:
                tensorboard.add_scalars(f"{self.name}/accuracy", {'train':correct/total * 100,'val':val_acc}, self.epoch_trained)
                tensorboard.add_scalars(f"{self.name}/loss", {'train':running_loss,'val': val_loss}, self.epoch_trained)
            
            


        if tensorboard:
            tensorboard.close()

    def _train_one_epoch(self, criterion, data):
        inputs, labels = data
        inputs = inputs.reshape(-1,1,300).double().to(self.cfg.device)
        labels = labels.to(self.cfg.device).long().to(self.cfg.device)
                
                # zero the parameter gradients
        self.optim.zero_grad()

                # forward + backward + optimize
        outputs = self(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        self.optim.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        return loss.item(), correct

    def predict(self, inputs):
        self.eval()
        inputs = inputs.reshape(-1,1,300).double()

        outputs = self(inputs)
        _, predicted = torch.max(outputs.data, 1)

        return predicted

class ResBlock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(ResBlock, self).__init__()
        self.expansion = 2
        self.conv1 = nn.Conv1d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.conv3 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)


        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        
        return x

class ResBlock(nn.Module):
    def __init__(
        self, in_channels, intermediate_channels, identity_downsample=None, stride=1
    ):
        super(ResBlock, self).__init__()
        self.expansion = 2
        self.conv1 = nn.Conv1d(
            in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn1 = nn.BatchNorm1d(intermediate_channels)
        self.conv2 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(intermediate_channels)
        self.conv3 = nn.Conv1d(
            intermediate_channels,
            intermediate_channels * self.expansion,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn3 = nn.BatchNorm1d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)


        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        
        return x

class ResNet(nn.Module):

    def __init__(self, num_classes=3, device="cpu", name="RestNet"):
        super(ResNet, self).__init__()
        self.in_channels = 1
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = nn.Sequential(ResBlock(16,8), ResBlock(16, 8))

        self.avgpool = nn.AdaptiveAvgPool1d(64)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16 * 64, num_classes)

        self.logsoftmax = nn.LogSoftmax(dim=1)

        self.optim = None
        self.name = name
        self.checkpoint = 0

        trained_epochs = 0

        self.device = device
        self.double()

        self.to(self.device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.logsoftmax(x)
        return x

    def save(self):
        torch.save(self.state_dict(), f'{MODELS_PATH}\\{self.name}_checkpoint_{self.checkpoint:03d}.model')

    def load(self, checkpoint='latest'):
        def get_checkpoint_number(path):
            return int(os.path.split(path)[1][-9:-6])

        if checkpoint == 'latest':
            models = list(glob.iglob(f"{MODELS_PATH}\\{self.name}_checkpoint_*.model"))
            model_path = max(models, key=get_checkpoint_number)
        else:
            model_path = f"{MODELS_PATH}\\{self.name}_checkpoint_{checkpoint:03d}.model"

        self.load_state_dict(torch.load(model_path))
        self.checkpoint = get_checkpoint_number(model_path)

    def _set_optim(self, fresh=False):
        if self.optim is None or fresh:
            self.optim = optim.Adam(self.parameters(), lr=0.001)

    def train_model(self, 
              train_loader, val_loader, epochs, 
              reset_optim=False,
              tensorboard=False,
              save_interval=None, 
              show=False,
              class_weights=None):

        self.train()
        self._set_optim(reset_optim)

        criterion = nn.NLLLoss(weight=class_weights) 

        if tensorboard:
            tensorboard = SummaryWriter(log_dir="C:/Users/luk/Desktop/kyselica/classification_of_light_curves/tensorboard/run")

        start_checkpoint = self.checkpoint

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            correct = 0
            total = 0

            for data in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}: ", leave=False):
                # get the inputs; data is a list of [inputs, labels]
                epoch_loss, epoch_correct = self._train_one_epoch(criterion, data)

                # print statistics
                running_loss += epoch_loss
                correct += epoch_correct
                total += data[1].size(0)

            running_loss /= total

            if save_interval and (epoch + 1) % save_interval == 0:
                self.save()
                self.checkpoint += 1

            val_acc, val_loss, _ = evaulate_net(self, val_loader)

            if show:
                print(f"Train:\n\tLoss: {running_loss}\n\tAcc: {correct/total*100}", flush=True)
                print(f"Validation:\n\tLoss: {val_loss}\n\tAcc: {val_acc}", flush=True)

            if tensorboard:
                tensorboard.add_scalars(f"{self.name}_{start_checkpoint:04d}/accuracy", {'train':correct/total * 100,'val':val_acc}, epoch)
                tensorboard.add_scalars(f"{self.name}_{start_checkpoint:04d}/loss", {'train':running_loss,'val': val_loss}, epoch)

        if tensorboard:
            tensorboard.close()

    def _train_one_epoch(self, criterion, data):
        inputs, labels = data
        inputs = inputs.reshape(-1,1,300).double().to(self.device)
        labels = labels.long().to(self.device)
                
                # zero the parameter gradients
        self.optim.zero_grad()

                # forward + backward + optimize
        outputs = self(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        self.optim.step()

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()

        return loss.item(), correct

    def predict(self, inputs):
        self.eval()
        inputs = inputs.reshape(-1,1,300).double().to(self.device)

        outputs = self(inputs)
        _, predicted = torch.max(outputs.data, 1)

        return predicted


def evaulate_net(net: Net, data_loader: DataLoader):
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.NLLLoss()

    pred_y = np.empty((0,0))
    true_y = np.empty((0,0))

    with torch.no_grad():   
        net.to(net.device)
        for data in data_loader:
            inputs, labels = data
            labels = labels.long().to(net.device)
            
            inputs = inputs.reshape(-1,1,300).double().to(net.device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            total_loss += loss.item()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if net.device != "cpu":
                predicted = predicted.detach().to('cpu')
                labels = labels.detach().to('cpu')
            pred_y = np.append(pred_y, predicted.numpy())
            true_y = np.append(true_y, labels.numpy())

    
        conf_matrix = confusion_matrix(true_y, pred_y)

        return 100 * correct / total, total_loss/total, conf_matrix

