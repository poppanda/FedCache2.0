import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ClientDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.synthesized_data = None
        self.synthesized_label = None
        self.transform = transform
        self._distribution = None

    def add_synthesized_data(self, new_data, new_labels):
        if isinstance(new_data, torch.Tensor):
            new_data = new_data.to('cpu').detach()
            new_labels = new_labels.to('cpu').detach()
        elif isinstance(new_data, np.ndarray):
            new_data = torch.tensor(new_data)
            new_labels = torch.tensor(new_labels)
        if self.synthesized_data is None:
            self.synthesized_data = new_data
            self.synthesized_label = new_labels
        else:
            self.synthesized_data = torch.cat((self.synthesized_data, new_data), dim=0)
            self.synthesized_label = torch.cat((self.synthesized_label, new_labels), dim=0)

    def clear_synthesized_data(self):
        self.synthesized_data = None
        self.synthesized_label = None

    def bound_synthesized_data_num(self, max_num):
        self.synthesized_data = self.synthesized_data[:max_num]
        self.synthesized_label = self.synthesized_label[:max_num]

    def get_distribution(self, num_classes):
        if self._distribution is None:
            self._distribution = np.bincount(self.labels, minlength=num_classes).astype(np.float32)
            self._distribution /= self._distribution.sum()
        return self._distribution

    def __getitem__(self, index):
        if self.synthesized_data is not None and index >= len(self.data):
            index -= len(self.data)
            data, label = self.synthesized_data[index], self.synthesized_label[index]
        else:
            data, label = self.data[index], self.labels[index]
        if self.transform:
            data = self.transform(data)
        return data, label

    def __len__(self):
        return len(self.labels) + (0 if self.synthesized_data is None else len(self.synthesized_data))
