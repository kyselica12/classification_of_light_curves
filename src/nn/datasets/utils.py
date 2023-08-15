import numpy as np
from pydoc import locate

from src.config import DataConfig


def create_datasets(labeled_data, cfg:DataConfig):

    (X_train, Y_train), (X_test, Y_test) = split_data_to_test_validation(labeled_data, cfg.labels, cfg.number_of_training_examples_per_class, cfg.validation_split)

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

def split_data_to_test_validation(data, labels, k, split=0.1):
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