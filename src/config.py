# CONFIGURATION

PACKAGE_PATH = "C:/Users/Kyselica/Desktop/kyselica/classification_of_light_curves"

# NET PARAMETERS
NET_NAME = "best"
INPUT_SIZE = 300
N_CHANNELS = 10
N_CLASSES = 3
HID_DIM = 128
STRIDE = 2
KERNEL_SIZE = 5

CHECKPOINT = None
DEVICE = "cuda"

# DATA PARAMETERS
LOAD_DATA = True

DATA_PATH = PACKAGE_PATH + "/resources"
LABELS = ["falcon_9",
          "atlas_5",
          "delta_4",
          "cz-3"
]


# FILTER PARAMETERS
FILTER_DATA = True

N_BINS = 15
N_GAPS = 0
GAP_SIZE = 0
NON_ZERP_RATIO = 0.2
RMS_RATIO = 0.1

# AUGMENTATION PARAMETERS
AUGMENTATION_MIN_EXAMPLES = 4000
AUGMENTATION_MAX_NOISE = 0.2
AUGMENTATION_LEAVE_ORIGINAL = False
AUGMENTATION_NUM_GAPS = 1
AUGMENTATION_GAP_PROB = 0.1
AUGMENTATION_MIN_LEN = 1
AUGMENTATION_MAX_LEN = 3

# END

def save_config(name):

    filename = f"{PACKAGE_PATH}/output/configurations/{name}.config"
    config_file = f"{PACKAGE_PATH}/src/config.py"


    with open(filename, "w") as ou_f:
        with open(config_file, "r") as in_f:
            line = in_f.readline()
            while "# END" not in line:
                print(line.strip(), file=ou_f)

