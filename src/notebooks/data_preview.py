import tqdm
import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append('../..')
from src.data.data_processor import DataProcessor
from src.sweeps.sweep import Sweep, DATA_CONFIG
from src.configs import PACKAGE_PATH, DataConfig, FilterConfig, NetArchitecture, CNNConfig, NetConfig, LC_SIZE, ResNetConfig
from src.configs import DataType as DT, AugmentType as A, SplitStrategy as ST

data_cfg = DataConfig(
        path=f"{PACKAGE_PATH}/Fall_2021_csv",
        output_path=f"{PACKAGE_PATH}/resources/datasets",
        class_names=["cz_3", "falcon_9", "atlas_V",  "h2a", "globalstar"],
        regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R_B$',  r'H-2A.*', r'GLOBALSTAR.*'],
        validation_split=0.2,
        split_strategy=ST.TRACK_ID,
        number_of_training_examples_per_class=100_000,
        filter_config=FilterConfig(n_bins=30, n_gaps= 10, gap_size=5, rms_ratio= 0., non_zero_ratio=0.8),
        # filter_config=None,
        data_types=[DT.LC],
        wavelet_start_scale=1,
        wavelet_scales_step=1,
        wavelet_end_scale=5,
        wavelet_name= 'gaus1',
        lc_shifts = 0,
        convert_to_mag=False,
        train_augmentations=[A.SHIFT],
)


dp = DataProcessor(data_cfg)

if os.path.exists(f'{dp.output_path}/{dp.hash}'):
    dp.load_data_from_file()
else:
    dp.create_dataset_from_csv()
    dp.save_data()

print("Data loaded")
print(dp.data.keys())


output = dp.to_output_format(dp.data)[0]
print(len(output))
input()

def plot_lc(lc1, lc2):
    fig, axs = plt.subplots(2)
    t = np.linspace(0,1, num=300)
    axs[0].scatter(t[lc1 != 0],lc1[lc1 != 0])
    axs[1].scatter(t[lc2 != 0],lc2[lc2 != 0])
    return fig

os.makedirs('images', exist_ok=True)

start = 5000
for i in tqdm.tqdm(range(start, start+2000)):
    fig = plot_lc(dp.data[DT.LC][i], output[i])
    plt.savefig(f'images/lc_{i}.png')
    plt.close(fig)
    del fig



# while(True):
#     img = np.random.rand(32, 32, 3)
#     cv.imshow('image',img)
#     k = cv.waitKey(0)
#     if k == ord('q'):
#         cv.destroyAllWindows()
#         break





