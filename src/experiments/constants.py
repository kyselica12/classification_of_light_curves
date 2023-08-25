from src.config import CNNConfig, CNNFCConfig, FCConfig, FourierDatasetConfig

FOLDER_NAME = "Experiments"

EPOCHS = 200
SAVE_INTERVAL = 50
BATCH_SIZE = 128
SAMPLER = True
MAX_EXAMPLES = 10000

LOAD = False
SEED = None
CHECKPOINT = "latest"

BEST_CNN_CONFIG = CNNConfig(
    input_size=300,
    output_size=5,
    in_channels=2,
    conv_layers=[(5, 11, 3)],
    classifier_layers=[256]
)

BEST_CNN_DATASET = FourierDatasetConfig(
    lc=True,
    push_to_max=True,
    reconstructed_lc=True
)

BEST_FC_CONFIG = FCConfig(
    input_size=16,
    output_size=5,
    layers=[128,256,256]
)

BEST_FC_DATASET = FourierDatasetConfig(
    fourier=True,
    amplitude=True,
    rms=True
)

BEST_COMBINED_DATASET = FourierDatasetConfig(
    fourier=True,
    amplitude=True,
    rms=True,
    lc=True,
    push_to_max=True,
)

BEST_CNN_FC_CONFIG = CNNFCConfig(
    input_size=300 + 16 + 1 + 1,
    output_size=5,
    cnn_input_size=300,
    in_channels=1,
    cnn_layers=BEST_CNN_CONFIG.conv_layers,
    fc_output_dim=BEST_FC_CONFIG.layers[-1],
    classifier_layers=[128]
)