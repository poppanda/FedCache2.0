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
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
    
class TMDDataset(BasicDataset):
    def __init__(self, root):
        data = np.load(root + '/TMD/features.npy').astype(np.float32)
        for col_id in range(data.shape[1]):
            data[:, col_id] /= np.abs(data[:, col_id]).max()
            print(data[:, col_id].max(), data[:, col_id].min())
        labels = np.load(root + '/TMD/labels.npy')
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
        
     