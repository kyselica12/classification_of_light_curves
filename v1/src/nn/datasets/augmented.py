import random
import numpy as np

from src.nn.datasets.basic import BasicDataset


class AugmentedBalancedDataset(BasicDataset):

    def __init__(self, 
                    data, labels, 
                    roll=True, add_gaps=True, add_noise=True,
                    max_noise=None, 
                    num_gaps=None, min_gap_len=None, max_gap_len=None, gap_prob=None, 
                    use_original_data=True, min_num_examples=1000) -> None:

        self.max_noise = max_noise
        
        self.num_gaps = num_gaps
        self.gap_prob = gap_prob
        self.min_len = min_gap_len
        self.max_len = max_gap_len

        self.data = data
        self.labels = labels


        self._normalize_data()

        self.augment_data(use_original_data, min_num_examples, roll, add_gaps, add_noise)

        self._shuffle_data()

        self.data = self.data.astype(np.float32)


    def augment_data(self, leave_original, class_size, roll, add_gaps, add_noise):
        unique_labels = np.unique(self.labels)

        augmented_data = []
        augmented_labels = []
        for l in unique_labels:
            
            label_data = self.data[self.labels == l].copy()
            augmented_label_data = []
            if  leave_original:
                augmented_label_data = list(label_data)

            if roll or add_gaps or add_noise:
                i = 0
                while len(augmented_label_data) < class_size:
                    d = self._get_augmented(label_data[i], roll, add_gaps, add_noise)
                    augmented_label_data.append(d)
                    i = (i + 1) % len(label_data)

            augmented_data.extend(augmented_label_data)
            augmented_labels.extend([l]*len(augmented_label_data))
        
        self.data = np.array(augmented_data)
        self.labels = np.array(augmented_labels).astype(np.int32)
     
    def _shuffle_data(self):
        shuffle_indices = np.random.permutation(len(self.data))

        self.data = self.data[shuffle_indices]
        self.labels = self.labels[shuffle_indices]
        
    def _get_augmented(self, data, roll, add_gaps, add_noise):
        if add_noise:
            data = self._add_noise(data)
        if roll:
            data = self._roll(data)
        if add_gaps:
            data = self._add_gap(data)
        
        return data
    
    def _add_noise(self, data):

        noise = np.random.randn(*data.shape) * self.max_noise
        return data + noise
    
    def _roll(self, data):
        
        shift = random.randrange(max(data.shape))

        return np.roll(data, shift)
        # return data

    def _add_gap(self, data):
        data = data.copy()
        DATA_SHAPE = max(data.shape)
        for _ in range(self.num_gaps):
            if random.random() <= self.gap_prob:
                size = random.randrange(self.min_len, self.max_len + 1)
                index = random.randrange(max(data.shape))
                data[np.linspace(index, index + size - 1, num=size).astype(np.int32) - DATA_SHAPE] = 0

        return data
