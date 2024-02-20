import os

import tqdm
import numpy as np

from src.config import Config, PACKAGE_PATH, DataConfig, FCConfig, FilterConfig, NetConfig
from src.train import Trainer
from src.nn.networks.utils import get_new_net, load_net
from src.nn.datasets.fourier import FourierDataset
from src.experiments.constants import *

def get_default_cfg():
    data_config = DataConfig(
            path=f"{PACKAGE_PATH}/resources/Fall_2021_R_B_globalstar.csv",
            labels=["cz_3", "falcon_9", "atlas",  "h2a", "globalstar"],
            regexes=[r'CZ-3B.*', r'FALCON_9.*', r'ATLAS_[5|V]_CENTAUR_R\|B$',  r'H-2A.*', r'GLOBALSTAR.*'],
            convert_to_mag=False,
            batch_size=BATCH_SIZE,
            number_of_training_examples_per_class = MAX_EXAMPLES,
            validation_split = 0.1,
            dataset_class="FourierDataset",
            dataset_arguments={},
            filter=FilterConfig(
                n_bins= 30,
                n_gaps= 10,
                gap_size= 5, 
                rms_ratio= 0.,
                non_zero_ratio=0.8
            )
    )

    net_cfg = NetConfig(
            name="Default",
            save_path=f"{PACKAGE_PATH}/output/models/{FOLDER_NAME}/",
            net_class="FC",  
            model_config=FCConfig(input_size=16, output_size=5, layers=[])  
    )

    cfg = Config(net_config=net_cfg, data_config=data_config)

    return cfg

    

def create_ouput_folders(folder_name):
    for f in ["models", "datasets","configurations"]:
        os.makedirs(f"{PACKAGE_PATH}/output/{f}/{folder_name}", exist_ok=True)

def load_dataset_to_trainer(trainer: Trainer, folder_name, cfg: Config):
    dataset_path = f"{PACKAGE_PATH}/output/datasets/{folder_name}"

    dataset_name = cfg.data_config.dataset_class + "_".join([f"{k}_{v}" for k,v in cfg.data_config.dataset_arguments.items()])
    dataset_name += f"_{MAX_EXAMPLES}"


    if os.path.exists(f"{dataset_path}/{dataset_name}"):
        trainer.load_data_from_file(f"{dataset_path}/{dataset_name}", cfg.data_config)
    else:
        trainer.load_data(cfg.data_config)
        os.makedirs(f"{dataset_path}/{dataset_name}", exist_ok=True)
        trainer.save_data(f"{dataset_path}/{dataset_name}")


def run(folder_name, 
        cfg: Config,
        epochs, epoch_save_interval, batch_size, sampler, output_csv_path,
        load, seed, checkpoint):

    create_ouput_folders(folder_name)
    
    trainer = Trainer(None, cfg.net_config, device=cfg.device)

    load_dataset_to_trainer(trainer, folder_name, cfg)

    if load:
        trainer.net = load_net(cfg.net_config, seed=seed, checkpoint=checkpoint)
    else:
        trainer.net = get_new_net(cfg, f"{PACKAGE_PATH}/output/configurations/{folder_name}/{cfg.net_config.name}.json")

    if  sampler:
        trainer.add_sampler()

    for _ in range(0,epochs, epoch_save_interval):
        trainer.train(epoch_save_interval, batch_size, tensorboard_on=True, save_interval=None, print_on=False)
        trainer.performance_stats(cfg.data_config.labels, save_path=output_csv_path)

def test_SDLCD(path, cfg, seed, epoch,checkpoint, train_dataset_folder, output_path='.'):

    trainer = Trainer(None, cfg.net_config, device=cfg.device)
    load_dataset_to_trainer(trainer, train_dataset_folder, cfg)
    trainer.net = load_net(cfg, seed=seed, epoch=epoch, checkpoint=checkpoint)
    trainer.net.to(cfg.device)
    trainer.net.double()

    test_set = FourierDataset([],[],**cfg.data_config.dataset_arguments)
    test_set.data = np.load(f"{path}/test_x.npy")
    test_set.labels = np.load(f"{path}/test_y.npy").astype(dtype=np.int32)

    trainer.val_set = test_set

    trainer.performance_stats(cfg.data_config.labels, save_path=f'{output_path}/SDLCD_test.csv')
    


def run_experiment(options, action, name):
    
    cfg = get_default_cfg()
    
    for op in tqdm.tqdm(options):
        print("Options: ", op)
        action(op, cfg)

        run(FOLDER_NAME,
            cfg,
            epochs=EPOCHS,
            epoch_save_interval=SAVE_INTERVAL,
            batch_size=cfg.data_config.batch_size,
            sampler=SAMPLER,
            output_csv_path=f"{PACKAGE_PATH}/output/{name}_results.csv",
            load=LOAD,
            seed=SEED,
            checkpoint=CHECKPOINT
        )
