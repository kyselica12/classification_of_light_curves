import os

from src.config import Config, PACKAGE_PATH
from src.train import Trainer
from src.nn.utils import get_new_net, load_net

def create_ouput_folders(folder_name):
    for f in ["models", "datasets","configurations"]:
        os.makedirs(f"{PACKAGE_PATH}/output/{f}/{folder_name}", exist_ok=True)

def load_dataset_to_trainer(trainer: Trainer, folder_name, cfg: Config):
    dataset_path = f"{PACKAGE_PATH}/output/datasets/{folder_name}"

    dataset_name = f"{cfg.data_config.filter.n_bins}_{cfg.data_config.filter.n_gaps}_{cfg.data_config.filter.gap_size}_{int(cfg.data_config.filter.non_zero_ratio * 10)}_{cfg.data_config.number_of_training_examples_per_class}"

    if os.path.exists(f"{dataset_path}/{dataset_name}"):
        trainer.load_data_from_file(f"{dataset_path}/{dataset_name}")
    else:
        trainer.load_data(cfg.data_config)
        os.makedirs(f"{dataset_path}/{dataset_name}", exist_ok=True)
        trainer.save_data(f"{dataset_path}/{dataset_name}")


def run(folder_name, 
        cfg: Config,
        epochs, epoch_save_interval, batch_size, sampler, output_csv_path,
        load, seed, checkpoint):

    create_ouput_folders(folder_name)
    
    trainer = Trainer(None)

    load_dataset_to_trainer(trainer, folder_name, cfg)

    if load:
        trainer.net = load_net(cfg, seed=seed, checkpoint=checkpoint)
    else:
        trainer.net = get_new_net(cfg, f"{PACKAGE_PATH}/output/configurations/{folder_name}/{cfg.net_config.name}.json")

    if  sampler:
        trainer.add_sampler()

    for i in range(0,epochs, epoch_save_interval):
        trainer.train(epoch_save_interval, batch_size, tensorboard_on=True, save_interval=None, print_on=False)
        trainer.evaulate(cfg.data_config.labels, save_path=output_csv_path)
    