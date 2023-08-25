import sys

sys.path.append("./")


from src.config import CNNConfig, CNNFCConfig, Config, DataConfig, FilterConfig, PACKAGE_PATH, FourierDatasetConfig, NetConfig, FCConfig, ConvLayer
from src.experiments.utils import run
import tqdm

FOLDER_NAME = "Experiments"#"Fourier_FC_8_8_v1"
EXPERIMENT_NAME = "CNNFC_Residuals"

EPOCHS = 500
SAVE_INTERVAL = 50
BATCH_SIZE = 128
SAMPLER = True
OUTPUT_CSV_PATH = f'{PACKAGE_PATH}/output/{EXPERIMENT_NAME}_results.csv'

LOAD = False
SEED = None
CHECKPOINT = "latest"

dataset_name = "FourierDataset"
dataset_args = FourierDatasetConfig(
    fourier=True,
    residuals=True,
    amplitude=True
).__dict__



net_class="CNNFC"
model_config=CNNFCConfig(
    input_size=16+1+300,
    output_size=5,
    in_channels=1,
    cnn_input_size=300,
    cnn_layers=[(5, 11,5), (20, 3,1)],
    fc_output_dim=256,
    fc_layers=[128,128],
    classifier_layers=[256]
)



net_config = NetConfig(
        name=f"{EXPERIMENT_NAME}",
        save_path=f"{PACKAGE_PATH}/output/models/{FOLDER_NAME}/",
        net_class=net_class,  
        model_config=model_config
)

data_config = DataConfig(
        path=f"{PACKAGE_PATH}/resources/Fall_2021_R_B_globalstar.csv",
        labels=["cz_3", "falcon_9", "atlas",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R\|B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        convert_to_mag=False,
        number_of_training_examples_per_class = 2000,
        validation_split = 0.1,
        dataset_class=dataset_name,
        dataset_arguments=dataset_args,
        filter=FilterConfig(
            n_bins= 30,
            n_gaps= 10,
            gap_size= 5, 
            rms_ratio= 0.,
            non_zero_ratio= 0.8
        )
)

cfg = Config(net_config=net_config, data_config=data_config)

OPTIONS = [
    (FourierDatasetConfig(
    fourier=True,
    residuals=True,
    amplitude=True
    ).__dict__
    , "Residuals"),
    (FourierDatasetConfig(
    fourier=True,
    amplitude=True,
    lc=True
    ).__dict__
    , "LC"),
    (FourierDatasetConfig(
    fourier=True,
    amplitude=True,
    reconstructed_lc=True
    ).__dict__
    , "Reconstructed LC"),
]

for op, name in tqdm.tqdm(OPTIONS):
    cfg.data_config.dataset_arguments = op
    cfg.net_config.name = f"{EXPERIMENT_NAME}_{name}"

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