import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


sys.path.append("./src")
sys.path.append("./")

from src.config import  PACKAGE_PATH, FourierDatasetConfig, FCConfig
from src.experiments.utils import run_experiment
from src.experiments.constants import *
from src.experiments.utils import get_default_cfg, load_dataset_to_trainer
from src.train import Trainer
from src.nn.networks.utils import get_new_net, load_net, save_net
from src.nn.datasets.fourier import FourierDataset


cfg = get_default_cfg()
cfg.data_config.dataset_class = "FourierDataset"
cfg.data_config.dataset_arguments = FourierDatasetConfig(fourier=True).__dict__
cfg.net_config.name = "FC_trained"
cfg.net_config.save_path = f"{PACKAGE_PATH}/output/models/Fourier_trained"

trainer = Trainer(None, cfg.net_config, cfg.device)

load_dataset_to_trainer(trainer, f"Experiments_Fourier", cfg)
trainer.train_set.offset = 16
trainer.val_set.offset = 16
trainer.train_set.compute_std_mean()

data = trainer.train_set.data
lables = trainer.train_set.labels

N = 5_000
indices = np.random.choice(data.shape[0], N, replace=False)
reduced_data = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=30).fit_transform(data[indices])
reduced_labels = lables[indices]


fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot()
# ax = fig.add_subplot(projection='3d')

for l in np.unique(reduced_labels):
    ok = reduced_labels == l
    # create 3D scatter plot
    plt.scatter(reduced_data[ok, 0], reduced_data[ok, 1], label=cfg.data_config.labels[l])
    # ax.scatter(reduced_data[ok, 0], reduced_data[ok, 1], reduced_data[ok, 2], label=cfg.data_config.labels[l])
plt.legend()
plt.show()