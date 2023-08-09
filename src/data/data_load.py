from collections import defaultdict
from typing import Callable, Dict, List
import pandas as pd
import numpy as np
import tqdm
import glob
import os
import re

from collections import defaultdict

from src.data.load_multi_array import load_multi_array

def load_data_multi_array(path, labels, convert_to_mag=False):
    data = defaultdict(list)
        
    files = [p for p in glob.iglob(f"{path}/*_multi_array.npy")]

    for file in tqdm.tqdm(files, desc=f"Folder {path}"):
        object_name = os.path.split(file)[1][:-len("_multi_array.npy")]
        label = get_object_label(object_name, labels)
        if label:
            arr = load_multi_array(file)
            if convert_to_mag:
                arr = -2.5 * np.log10(arr)
            data[label].extend(arr)

    for key in data:
        data[key] = np.array(data[key])

    return data

def load_data(path, labels, regexes=None, convert_to_mag=False):
    data = defaultdict(list)
        
    files = [p for p in glob.iglob(f"{path}/*.npy")]

    for file in tqdm.tqdm(files, desc=f"Folder {path}"):
        object_name = os.path.split(file)[1][:-len(".npy")]
        label = get_object_label(object_name, labels, regexes)
        if label:
            arr = np.load(file)
            # if np.any(arr < 0):   # Magnitute can be negative
            #     arr += np.abs(np.min(arr)) + 0.000000001
            if convert_to_mag:
                arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])
            print(f"Label: {label} {len(arr)} examples.")
            data[label] = arr

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

def get_object_label(name, labels, regexes=None):
    name2 = name.lower().replace("_", "").replace("-", "")
    for i, label in enumerate(labels):
        label2 = label.lower().replace("_", "").replace("-", "")

        if regexes is not None:
            if re.search(regexes[i], name, re.IGNORECASE):
                return label            
        else:
            if label2 in name2.lower() and not "deb" in name2.lower():
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
    
def load_data_from_numpy_arrays(path):
    data = {}
    for filepath in glob.iglob(f"{path}/*.npy"):
        label = os.path.split(filepath)[1][:-len(".npy")]
        data[label] = np.load(filepath)

    return data  
    