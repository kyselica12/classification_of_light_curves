import sys
sys.path.append("./")


from src.config import Config, DataConfig, FilterConfig, PACKAGE_PATH, FourierDatasetConfig, NetConfig, FCConfig
from src.experiments.utils import run
import tqdm

FOLDER_NAME = "new_dataset_fc"#"Fourier_FC_8_8_v1"
EXPERIMENT_NAME = "Architecture"

EPOCHS = 1
SAVE_INTERVAL = 1
BATCH_SIZE = 64
SAMPLER = True
OUTPUT_CSV_PATH = f'{PACKAGE_PATH}/output/{EXPERIMENT_NAME}_results.csv'

LOAD = False
SEED = None
CHECKPOINT = "latest"

net_config = NetConfig(
        name=f"{EXPERIMENT_NAME}",
        net_class="FC",
        input_size=17 + 17 + 1,
        n_classes=5,
        device="cuda:0",
        save_path=f"{PACKAGE_PATH}/output/models/{FOLDER_NAME}/",
        net_args=FCConfig(
            layers=[256]
        ).__dict__
)

data_config = DataConfig(
        path=f"{PACKAGE_PATH}/resources/Fall_2021_R_B_globalstar.csv",
        from_csv=True,
        labels=["cz_3", "falcon_9", "atlas",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R\|B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        convert_to_mag=False,
        number_of_training_examples_per_class = 2000,
        validation_split = 0.1,
        dataset_class="FourierDataset",
        dataset_arguments=FourierDatasetConfig(
        ).__dict__,
        filter=FilterConfig(
            n_bins= 30,
            n_gaps= 10,
            gap_size= 5, 
            rms_ratio= 0.,
            non_zero_ratio= 0.8
        )
)

data_config.save_path = f"Fourier_params_{data_config.number_of_training_examples_per_class}_{data_config.validation_split}"

cfg = Config(net_config=net_config, data_config=data_config)


NET_CONFIGS = [
    [128],
    [256],
    [512],
    [128, 128],
    [128, 256],
    [128, 256, 512],
    [128, 256, 128],
    [256, 512, 256],
    [128, 256, 512, 256]
    [128, 256, 512, 256, 128]
]


for i, net_config in tqdm.tqdm(enumerate(NET_CONFIGS)):
    cfg.net_config.net_args = FCConfig(layers=net_config).__dict__
    cfg.net_config.name = f"{EXPERIMENT_NAME}_{i}"

    run(FOLDER_NAME,
        cfg,
        epochs=EPOCHS,
        epoch_save_interval=SAVE_INTERVAL,
        batch_size=BATCH_SIZE,
        sampler=SAMPLER,
        output_csv_path=OUTPUT_CSV_PATH,
        load=LOAD,
        seed=SEED,
        checkpoint=CHECKPOINT
        )
