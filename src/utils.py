import numpy as np
from pytorch_lightning import Trainer
import wandb
import torch

from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.loggers import WandbLogger

from src.configs import WANDB_KEY_FILE, DataConfig
from src.data.data_processor import DataProcessor
from src.module.lightning_module import LCModule

WANDB_API_KEY = None
with open(WANDB_KEY_FILE, 'r') as f:
    WANDB_API_KEY = f.read().strip()

def log_in_to_wandb():
    try:
        wandb.login(key=WANDB_API_KEY)
    except:
        print('To use your W&B account,\nGo to Add-ons -> Secrets and provide your W&B access token. Use the Label name as WANDB. \nGet your W&B access token from here: https://wandb.ai/authorize')

def get_wandb_logger(project, name=None):
    log_in_to_wandb()
    return WandbLogger(project=project, name=name, log_model=False)


def train(module: LCModule,
          dp: DataProcessor,
          num_epochs: int = 10,
          batch_size: int = 32,
          num_workers: int = 4, 
          callbacks: list = [],
          sampler=False,
          max_num_samples=10**6,
          logger = None):

    if logger is not None and isinstance(logger, WandbLogger):
        logger.log_hyperparams({"data": dp.data_config.__dict__})
        module.log_confusion_matrix = True

    
    train_set, val_set, test_set = dp.get_pytorch_datasets()
    
    wr_sampler = None
    if sampler:
        labels_unique, counts = np.unique(train_set.labels, return_counts=True)
        weights = [sum(counts) / c  for c in counts]
        example_weights = [weights[int(l)] for l in train_set.labels]
        wr_sampler = WeightedRandomSampler(torch.DoubleTensor(example_weights), max_num_samples)

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              sampler=wr_sampler,
                              pin_memory=True,
                              shuffle=not sampler,
                              drop_last=True)

    val_loader = DataLoader(val_set, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              drop_last=True)
    
    test_loader = None
    if test_set is not None:
        test_loader = DataLoader(test_set, 
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False,
                                drop_last=True)
        # val_loaders.append(test_loader)
        
    
    trainer = Trainer(default_root_dir='TODO', #TODO: Better default root dir
                      max_epochs=num_epochs,
                      logger=logger,
                      callbacks=callbacks)
    
    trainer.fit(module, train_loader, val_loader)

    trainer.test(module, test_loader)

    if logger is not None and isinstance(logger, WandbLogger):
        wandb.finish()
