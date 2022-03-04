from collections import defaultdict
from typing import Callable, Dict, List
import numpy as np
import tqdm
import glob
import os
from collections import defaultdict

from data.load_multi_array import load_multi_array
from config import DataConfig

def load_data(cfg: DataConfig):
    data = defaultdict(list)
        
    files = [p for p in glob.iglob(f"{cfg.path}/*_multi_array.npy")]

    for file in tqdm.tqdm(files, desc=f"Folder {cfg.path}"):
        object_name = os.path.split(file)[1][:-len("_multi_array.npy")]
        label = get_object_label(object_name, cfg.labels)
        if label:
            arr = load_multi_array(file)
            data[label].extend(arr)

    for key in data:
        data[key] = np.array(data[key])

    return data

def load_all_data(path: str) -> Dict[str, np.array]:
    data = defaultdict(list)
        
    files = [p for p in glob.iglob(f"{path}/*_multi_array.npy")]

    for file in tqdm.tqdm(files, desc=f"Folder {path}"):
        object_name = os.path.split(file)[1][:-len("_multi_array.npy")]
        arr = load_multi_array(file)
        data[object_name].extend(arr)

    for key in data:
        data[key] = np.array(data[key])

    return data

def get_labeled_data(data: Dict[str, np.array], labels: List[str]) -> Dict[str, np.array]:

    labeled_data = defaultdict(lambda: np.empty((0,300)))

    for name in data:
        label = get_object_label(name, labels)
        if label:
            labeled_data[label] = np.append(labeled_data[label], data[name], axis=0)
    
    return labeled_data

def get_object_label(name, labels):
    for label in labels:
        if label in name.lower():
            return label
    return None

def get_non_zero_ratio(data: np.array) -> np.array:
    return np.sum(data != 0, axis=1)

def dict_map(data: Dict, func: Callable) -> Dict:
    return { key: func(value) for key, value in data.items()}

def sort_func(arr):
    stats = np.sum(arr != 0, axis=1)
    indices = np.argsort(-stats)
    return arr[indices]

def get_representants(arr):

    best = arr[:3]
    worst = arr[-3:]

    mid = len(arr)//2
    middle = arr[mid-1:mid+2]

    data = np.concatenate((best, middle, worst))
    titles = [f'Signal ratio: {x/300 * 100:.2f}%' for x in np.sum(data != 0, axis=1)]

    return data, titles

def get_stats(arr):
    stats = np.sum(arr != 0, axis=1) / 300 * 100
    return stats
    