import argparse
import sys
from typing import List

import wandb

sys.path.append(".")


from src.utils import get_wandb_logger, train
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, ModelConfig, NetArchitecture, FCConfig, CNNConfig, NetConfig
from src.data.data_processor import DataProcessor
from src.module.lightning_module import LCModule

def parse_args():
    parser = argparse.ArgumentParser(description='Train the model')

    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--arr', type=List[int], default=[])
    return parser.parse_args()

def test(config):
    print(config.epochs)
    print(config.batch_size)
    print(config.arr)
    wandb.log({"test": 1})

if __name__ == "__main__":

    config = parse_args()

    test(config)

    
    # PROJECT = "SWEEP_TEST"
    # NAME = "NAME"
    # EPOCHS = args.epochs
    # BATCH_SIZE = args.batch_size
    # NUM_WORKERS = 4


    # filter=FilterConfig(
    #     n_bins= 30,
    #     n_gaps= 10,
    #     gap_size= 5, 
    #     rms_ratio= 0.,
    #     non_zero_ratio=0.8
    # )

    # data_config = DataConfig(
    #     path="/mnt/c/work/Fall_2021_csv",
    #     output_path=f"{PACKAGE_PATH}/resources/datasets",
    #     class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
    #     regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
    #     validation_split=0.2,
    #     split_strategy="objectID",
    #     number_of_training_examples_per_class=10_000,
    #     filter_config=filter,
    #     data_types=[],
    #     convert_to_mag=False,
    #     wavelet_scales= 10,
    #     wavelet_name= 'gaus1'
    # )
    # dp = DataProcessor(data_config)

    # module_config = NetConfig(
    #     name="default",
    #     input_size=dp.data_shape[0],
    #     class_names=data_config.class_names,
    #     output_size=5,
    #     architecture=NetArchitecture.FC,
    #     args=FCConfig(input_size=16, output_size=5, layers=[])  
    # )

    # dp.load_data_from_file()
    # module = LCModule(module_config)
    # logger = get_wandb_logger(PROJECT, NAME)

    # train(module, dp,
    #     num_epochs=EPOCHS,
    #     batch_size=BATCH_SIZE,
    #     num_workers=NUM_WORKERS,
    #     logger=logger)


