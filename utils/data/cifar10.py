import logging
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .client_dataset import ClientDataset
from .basic_dataset import BasicDataset
from .tools import image_to_numpy, numpy_collate

logger = logging.getLogger(__name__)


class CIFAR10Dataset(BasicDataset):
    def __init__(self, root, image_size=32, extend_size=8):
        self.img_size = image_size
        self.train_transform = transforms.Compose([
            # transforms.Resize(size=image_size + extend_size, antialias=True),
            # transforms.RandomResizedCrop(size=image_size),
            image_to_numpy])
        self.test_transform = image_to_numpy
        train_data = datasets.CIFAR10(root=root, train=True, download=True, transform=self.train_transform)
        test_data = datasets.CIFAR10(root=root, train=False, download=True, transform=self.test_transform)
        train_data.data = torch.tensor(train_data.data, dtype=torch.float32)
        train_data.targets = torch.tensor(train_data.targets, dtype=torch.long)
        test_data.data = torch.tensor(test_data.data, dtype=torch.float32)
        test_data.targets = torch.tensor(test_data.targets, dtype=torch.long)
        super().__init__(train_data, test_data)

        self.class_names = self.train_data.classes
    def get_dataloader(self, batch_size, num_workers, shuffle=True):
        return DataLoader(self.train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                          collate_fn=numpy_collate, pin_memory=True)