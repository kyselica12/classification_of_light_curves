import sys

sys.path.append("./")


from src.config import CNNConfig, CNNFCConfig, Config, DataConfig, FilterConfig, PACKAGE_PATH, FourierDatasetConfig, NetConfig, FCConfig, ConvLayer
from src.experiments.utils import run
import tqdm

FOLDER_NAME = "test"#"Fourier_FC_8_8_v1"
EXPERIMENT_NAME = "TEST"

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
        net_class="CNNFC",  
        save_path=f"{PACKAGE_PATH}/output/models/{FOLDER_NAME}/",
        model_config=CNNFCConfig(
            input_size=300,
            output_size=5,
            in_channels=1,
            cnn_input_size=200,
            cnn_layers=[ConvLayer(10,3), ConvLayer(20,3)],
            fc_output_dim=10,
            fc_layers=[64],
            classifier_layers=[64]
        )
)

data_config = DataConfig(
        path=f"{PACKAGE_PATH}/resources/Fall_2021_R_B_globalstar.csv",
        labels=["cz_3", "falcon_9", "atlas",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R\|B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        convert_to_mag=False,
        number_of_training_examples_per_class = 10000,
        validation_split = 0.1,
        dataset_class="FourierDataset",
        dataset_arguments=FourierDatasetConfig(
            lc=True,
        ).__dict__,
        filter=FilterConfig(
            n_bins= 30,
            n_gaps= 10,
            gap_size= 5, 
            rms_ratio= 0.,
            non_zero_ratio= 0.8
        )
)

cfg = Config(net_config=net_config, data_config=data_config)


NET_CONFIGS = [
    [128],
    # [256],
    # [512],
    # [128, 128],
    # [128, 256],
    # [128, 256, 512],
    # [128, 256, 128],
    # [256, 512, 256],
    # [128, 256, 512, 256],
    # [128, 256, 512, 256, 128]
]


for i, net_config in tqdm.tqdm(enumerate(NET_CONFIGS)):
    cfg.net_config.model_config.layers = net_config
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
