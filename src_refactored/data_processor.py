import re
from collections import defaultdict
import numpy as np
import pandas as pd

from src_refactored.configs import DataConfig


class DataProcessor:
    
    def __init(self, data_config: DataConfig):
        self.data_config = data_config
        self.labels = self.data_config.labels
        self.regexes = self.data_config.regexes
        self.path = self.data_config.path
        self.output_path = self.data_config.output_path
        self.validation_split = self.data_config.validation_split
        self.number_of_training_examples_per_class = self.data

    def _load_csv(self):
        
        df = pd.read_csv(self.path)
        data_dict = defaultdict(list)

        for name in df["Object name"].unique():
            
            if (label := self.get_object_label(name, self.labels, self.regexes)) is None:
                continue

            object_data = df[df["object name"] == name]  
            object_IDs = object_data["Object ID"].unique()

            for object_ID in object_IDs:
                arr = object_data[object_data["Object ID"] == object_ID].to_numpy()[:, 4:] # remove first 4 columns

                if self.convert_to_mag:
                    arr[arr != 0] = -2.5 * np.log10(arr[arr != 0])
                data_dict[label].append(arr)
            
        return data_dict

    def get_object_label(self, name, labels, regexes=None):
        def remove_extra_chars(string):
            return string.lower().replace("_", "").replace("-", "")

        if regexes:
            return [l for l, r in zip(labels, regexes) if re.search(r, name, re.IGNORECASE)][0]

        return [l for l in labels if remove_extra_chars(l) in remove_extra_chars(name) and not "deb" in remove_extra_chars(name)][0]



        