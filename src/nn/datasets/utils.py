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

    if cfg.from_csv:
        (X_train, Y_train), (X_test, Y_test) = split_all_data_to_test_validation_by_object(labeled_data, cfg.labels, cfg.number_of_training_examples_per_class, cfg.validation_split)
    else:
        (X_train, Y_train), (X_test, Y_test) = split_data_to_test_validation(labeled_data, cfg.labels, cfg.number_of_training_examples_per_class, cfg.validation_split)

    idx_train, idx_test = np.random.permutation(len(X_train)), np.random.permutation(len(X_test))

    X_train, X_test = X_train[idx_train], X_test[idx_test]
    Y_train, Y_test = Y_train[idx_train], Y_test[idx_test]

    DatasetClass = find_dataset_class(cfg.dataset_class)
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

def split_object_data_to_test_validation(data, label, k, split=0.1):

    sizes = [len(i) for i in data[label]]

    N = sum(sizes)

    indices = np.argsort(-np.array(sizes))
    
    total = 0
    train = np.empty((0, *data[label][0].shape[1:]))
    val = np.empty((0, *data[label][0].shape[1:]))
    
    for i in range(len(indices)):
        if (sizes[indices[i]] + total < k*1.1 and sizes[indices[i]] + total < N * (1-split)) or \
           (total == 0 and sizes[indices[i]] + total < N * (1-split)):
            total += sizes[indices[i]]
            train = np.concatenate((train, data[label][indices[i]]))
        else:
            val = np.concatenate((val, data[label][indices[i]]))

   
    return train, val

def split_all_data_to_test_validation_by_object(data, labels, k, split=0.1):
    X_train, X_val = None, None
    Y_train, Y_val = None, None
    for i, label in enumerate(labels):
        obj_train, obj_val = split_object_data_to_test_validation(data, label, k, split)
        print(f"\n{label:15}: {len(obj_train):5} training examples, {len(obj_val):5} validation examples")
        
        if X_train is None:
            X_train = obj_train
            X_val = obj_val
            Y_train = np.array([i]*len(obj_train))
            Y_val = np.array([i]*len(obj_val))
        else:
            X_train = np.concatenate((X_train, obj_train))
            X_val = np.concatenate((X_val, obj_val))
            Y_train = np.concatenate((Y_train, np.array([i]*len(obj_train))))
            Y_val = np.concatenate((Y_val, np.array([i]*len(obj_val))))

    id_train = np.random.permutation(len(X_train))
    id_val = np.random.permutation(len(X_val))

    X_train, Y_train = X_train[id_train], Y_train[id_train]
    X_val, Y_val = X_val[id_val], Y_val[id_val]

    return (X_train, Y_train), (X_val, Y_val)


def find_dataset_class(class_name):
    package_name = class_name.lower()[:-len("dataset")]
    return locate(f'src.nn.datasets.{package_name}.{class_name}')