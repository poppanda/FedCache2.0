import os
import torch
import numpy as np
from PIL import Image
from .tools import image_to_numpy
from .basic_dataset import BasicDataset
from torchvision import datasets, transforms
from tqdm import tqdm


class RawData:
    def __init__(self, data, targets, classes):
        self.data = data
        self.targets = targets
        self.classes = classes
    
    def __len__(self):
        return len(self.data)


def load_data(root, labels, data_type='torch'):
    data, targets, classes = [], [], []
    for label in tqdm(labels, desc="load data in Cinic10", leave=False):
        label_id = labels[label]
        classes.append(label)
        if not os.path.exists(os.path.join(root, f"{label}.npz")):
            tmp_data, tmp_targets = [], []
            for file in os.listdir(os.path.join(root, label)):
                if file.endswith('.png'):
                    img = Image.open(os.path.join(root, label, file))
                    img_array = np.array(img)
                    if img_array.shape != (32, 32, 3):
                        continue
                    tmp_data.append(img_array)
                    tmp_targets.append(label_id)
            tmp_data = np.array(tmp_data, dtype=np.float32)
            tmp_targets = np.array(tmp_targets, dtype=int)
            np.savez_compressed(os.path.join(root, f"{label}.npz"), data=tmp_data, targets=tmp_targets)
        else:
            npz = np.load(os.path.join(root, f"{label}.npz"))
            tmp_data = npz['data']
            tmp_targets = npz['targets']
        data.append(tmp_data.astype(np.uint8))
        targets.append(tmp_targets.astype(int))
    data = np.concatenate(data)
    targets = np.concatenate(targets)
    if data_type == 'torch':
        data = torch.tensor(data, dtype=torch.float32)
        targets = torch.tensor(targets, dtype=torch.long)
    else:
        data = data.astype(np.float32)
        targets = targets.astype(int)
    return RawData(data, targets, classes)


class CINIC10Dataset(BasicDataset):
    def __init__(self, root, image_size=32, extend_size=8, data_type='torch'):
        self.img_size = image_size
        labels = {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4, 'dog': 5,
                  'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
        root = os.path.expanduser(root)
        train_data = load_data(root=os.path.join(root, 'cinic10', 'train'), labels=labels, data_type=data_type)
        test_data = load_data(root=os.path.join(root, 'cinic10', 'test'), labels=labels, data_type=data_type)

        train_data.data = train_data.data
        train_data.targets = train_data.targets
        test_data.data = test_data.data
        test_data.targets = test_data.targets
        super().__init__(train_data, test_data)
        self.class_names = self.train_data.classes
