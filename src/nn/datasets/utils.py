import random
from pydoc import locate

import numpy as np

from src.config import DataConfig

def split_data_to_test_validation(labeled_data, labels, max_number_of_training_examples=None, validation_split=0.1):
    X_test, X_train = [], []
    Y_test, Y_train = [], []

    labels_id = {l: i for i, l in enumerate(labels)}

    for obj in labeled_data:
        x = labeled_data[obj]
        N = int(len(x))
        y = [labels_id[obj]]*N
        
        if max_number_of_training_examples is None:
            max_number_of_training_examples = N
        
        n = int(N * (1 - validation_split))
        n = min(n, max_number_of_training_examples)

        random.shuffle(x)

        X_test.extend(x[n:])
        Y_test.extend(y[n:])

        X_train.extend(x[:n])
        Y_train.extend(y[:n])


    X_train, X_test = np.array(X_train), np.array(X_test)
    Y_train, Y_test = np.array(Y_train, dtype=np.int32), np.array(Y_test, dtype=np.int32)

    return (X_train, Y_train), (X_test, Y_test)

def create_datasets(labeled_data, cfg:DataConfig):

    (X_train, Y_train), (X_test, Y_test) = split_data_to_test_validation(labeled_data, cfg.labels, cfg.number_of_training_examples_per_class, cfg.validation_split)

    idx_train, idx_test = np.random.permutation(len(X_train)), np.random.permutation(len(X_test))

    X_train, X_test = X_train[idx_train], X_test[idx_test]
    Y_train, Y_test = Y_train[idx_train], Y_test[idx_test]

    DatasetClass = locate(f'src.nn.datasets.{cfg.dataset_class.lower()}.{cfg.dataset_class}')
    val_set = DatasetClass(X_test, Y_test, **cfg.dataset_arguments)
    train_set = DatasetClass(X_train, Y_train, **cfg.dataset_arguments)

    print(f"Training set: {len(train_set)}")
    print(f"Validation set: {len(val_set)}")

    if cfg.save_path:
        np.save(f"{cfg.save_path}/train_x.np", X_train)
        np.save(f"{cfg.save_path}/train_y.np", Y_train)
        np.save(f"{cfg.save_path}/test_x.np", X_test)
        np.save(f"{cfg.save_path}/test_y.np", Y_test)

    return train_set, val_set

def find_dataset_class(class_name):
    package_name = class_name.lower()[:-len("dataset")]
    return locate(f'src.nn.datasets.{package_name}.{class_name}')