import argparse
import sys

sys.path.append(".")


from src.utils import get_wandb_logger, train
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, ModelConfig, NetArchitecture, FCConfig, CNNConfig, NetConfig
from src.data.data_processor import DataProcessor
from src.module.lightning_module import LCModule

parser = argparse.ArgumentParser(description='Train the model')

parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

if __name__ == "__main__":

    args = parser.parse_args()

    PROJECT = "SWEEP_TEST"
    NAME = "NAME"
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = 4


    filter=FilterConfig(
        n_bins= 30,
        n_gaps= 10,
        gap_size= 5, 
        rms_ratio= 0.,
        non_zero_ratio=0.8
    )

    data_config = DataConfig(
        path="/mnt/c/work/Fall_2021_csv",
        output_path=f"{PACKAGE_PATH}/resources/datasets",
        class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        validation_split=0.2,
        split_strategy="objectID",
        number_of_training_examples_per_class=10_000,
        filter_config=filter,
        convert_to_mag=False,
        fourier=True,
        std= False,
        rms= False,
        residuals= False,
        amplitude= False,
        lc= False,
        reconstructed_lc= False,
        push_to_max= False,
        wavelet=False,
        wavelet_scales= 10,
        wavelet_name= 'gaus1'
    )

    module_config = NetConfig(
        name="default",
        input_size=16,
        class_names=data_config.class_names,
        output_size=5,
        architecture=NetArchitecture.FC,
        args=FCConfig(input_size=16, output_size=5, layers=[])  
    )

    dp = DataProcessor(data_config)
    dp.load_data_MMT()
    module = LCModule(module_config)
    logger = get_wandb_logger(PROJECT, NAME)

    train(module, dp,
        num_epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        logger=logger)


