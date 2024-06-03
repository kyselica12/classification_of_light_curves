import sys
import wandb

sys.path.append("../..")
sys.path.append("..")
sys.path.append(".")

from src.utils import train, log_in_to_wandb
# from src.sweeps.input_sweep import WaveletSweep as S
from src.sweeps.wavelet_sweep import WaveletSweep as S
# from src.sweeps.resnet_preprocessing_sweep import ResnetPreprocesingSweep as S

if __name__ == "__main__":

    sweep = S()

    log_in_to_wandb()
    sweep_id = wandb.sweep(sweep.get_wandb_sweep_cfg(), project="Preprocessing")    

    wandb.agent(sweep_id, function=sweep.run)