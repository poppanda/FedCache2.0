import logging
import torch
import numpy as np 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .client_dataset import ClientDataset
from .basic_dataset import BasicDataset

logger = logging.getLogger(__name__)

class RawDataset:
    def __init__(self, data, labels):
        self.data = torch.tensor(data)
        self.targets = torch.tensor(labels)
        self.classes = np.unique(labels)
        # print(self.classes)
        
    def __len__(self):
        return len(self.data)

class UrbanSoundDataset(BasicDataset):
    def __init__(self, root):
        data = np.load(root + '/UrbanSound8K/features.npy')
        labels = np.load(root + '/UrbanSound8K/labels.npy')
        # shuffle the data
        idx = np.random.permutation(len(data))
        data = data[idx]
        labels = labels[idx]
        train_data = data[:int(0.8*len(data))]
        train_labels = labels[:int(0.8*len(data))]
        test_data = data[int(0.8*len(data)):]
        test_labels = labels[int(0.8*len(data)):]
        self.train_data = RawDataset(train_data, train_labels)
        self.test_data = RawDataset(test_data, test_labels)


if __name__ == "__main__":
    UrbanSoundDataset('~/Code/data/')
