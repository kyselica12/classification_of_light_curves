from pytorch_lightning import Trainer
import wandb

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger

from src_refactored.configs import WANDB_API_KEY
from src_refactored.data_processor import DataProcessor
from src_refactored.lightning_module import LCModule

WANDB_API_KEY = None
with open(WANDB_API_KEY, 'r') as f:
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
          logger = None):

    if logger is not None and isinstance(logger, WandbLogger):
        logger.log_hyperparams({"data": dp.data_config.__dict__})
    
    train_set, val_set = dp.get_pytorch_datasets()

    train_loader = DataLoader(train_set, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=True,
                              drop_last=True)

    val_loader = DataLoader(val_set, 
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False,
                              drop_last=True)
    
    trainer = Trainer(default_root_dir='TODO',
                      max_epochs=num_epochs,
                      logger=logger,
                      callbacks=callbacks)
    
    trainer.fit(module, train_loader, val_loader)

    if logger is not None and isinstance(logger, WandbLogger):
        wandb.finish()
        
