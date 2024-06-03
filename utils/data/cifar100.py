import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .client_dataset import ClientDataset
from .basic_dataset import BasicDataset
from .tools import image_to_numpy, numpy_collate

logger = logging.getLogger(__name__)


class CIFAR100Dataset(BasicDataset):
    def __init__(self, root, image_size=32, extend_size=8):
        self.img_size = image_size
        self.train_transform = transforms.Compose([
            # transforms.Resize(size=image_size + extend_size, antialias=True),
            # transforms.RandomResizedCrop(size=image_size),
            image_to_numpy])
        self.test_transform = image_to_numpy
        train_data = datasets.CIFAR100(root=root, train=True, download=True, transform=self.train_transform)
        test_data = datasets.CIFAR100(root=root, train=False, download=True, transform=self.test_transform)
        superclass = [['beaver', 'dolphin', 'otter', 'seal', 'whale'],
                      ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
                      ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
                      ['bottle', 'bowl', 'can', 'cup', 'plate'],
                      ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
                      ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
                      ['bed', 'chair', 'couch', 'table', 'wardrobe'],
                      ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
                      ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
                      ['bridge', 'castle', 'house', 'road', 'skyscraper'],
                      ['cloud', 'forest', 'mountain', 'plain', 'sea'],
                      ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
                      ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
                      ['crab', 'lobster', 'snail', 'spider', 'worm'],
                      ['baby', 'boy', 'girl', 'man', 'woman'],
                      ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
                      ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
                      ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
                      ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
                      ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor']]
        cls_str_to_super_class = {cls: i for i, classes in enumerate(superclass) for cls in classes}
        cls_to_super_cls = dict()
        for i, cls_str in enumerate(train_data.classes):
            cls_to_super_cls[i] = cls_str_to_super_class[cls_str]
        train_data.data = torch.tensor(train_data.data, dtype=torch.float32)
        train_data.targets = torch.tensor(list(map(lambda x: cls_to_super_cls[x], train_data.targets)), dtype=torch.long)
        test_data.data = torch.tensor(test_data.data, dtype=torch.float32)
        test_data.targets = torch.tensor(list(map(lambda x: cls_to_super_cls[x], test_data.targets)), dtype=torch.long)
        super().__init__(train_data, test_data)
        self.class_names = self.train_data.classes
