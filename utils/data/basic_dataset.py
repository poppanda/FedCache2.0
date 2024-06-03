import torch
from torch import Tensor
import numpy as np
from typing import Tuple
from logging import getLogger
from .client_dataset import ClientDataset
from torch.utils.data import DataLoader, Dataset

logger = getLogger(__name__)


class BasicDataset:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data

    def transpose_data(self, target_index=(0, 3, 1, 2)):
        print(f"Before transpose: {self.train_data.data.shape}")
        if isinstance(self.train_data.data, np.ndarray):
            self.train_data.data = self.train_data.data.transpose(*target_index)
            self.test_data.data = self.test_data.data.transpose(*target_index)
        elif isinstance(self.train_data.data, Tensor):
            self.train_data.data = self.train_data.data.permute(*target_index)
            self.test_data.data = self.test_data.data.permute(*target_index)

    def scale_by(self, scale_factor):
        self.train_data.data = self.train_data.data * scale_factor
        self.test_data.data = self.test_data.data * scale_factor

    def get_heterogeneous_datasets(self, num_clients, num_classes, alpha, transform=None) -> list[ClientDataset]:
        raw_data = self.train_data.data
        raw_labels = self.train_data.targets
        if raw_data.mean() > 1:
            print("WARNING! raw_data avg > 1")
        client_idxes = [np.empty(0, dtype=np.int64) for _ in range(num_clients)]
        ret_datasets = []
        min_size = 0
        max_num_per_client = raw_data.shape[0] / num_clients
        while min_size < 1:
            for k in range(num_classes):
                np.random.seed(k)
                idx_k = np.where(raw_labels == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([cli_p * (1 if len(cli_idx) < max_num_per_client else 0)
                                        for cli_p, cli_idx in zip(proportions, client_idxes)])
                proportions /= proportions.sum()
                cum_proportions = np.cumsum(proportions * idx_k.shape[0]).astype(int)[:-1]
                for client_id, new_idx in zip(range(num_clients), np.split(idx_k, cum_proportions)):
                    client_idxes[client_id] = np.concatenate([client_idxes[client_id], new_idx])
            min_size = min([len(client_idx) for client_idx in client_idxes])
        logger.info(f"Minimum client dataset size: {min_size}")
        logger.info(f"Maximum client dataset size: {max([len(client_idx) for client_idx in client_idxes])}")
        logger.info(f"Total client dataset size: {sum([len(client_idx) for client_idx in client_idxes])}")
        for client_idx in client_idxes:
            client_data = raw_data[client_idx]
            client_labels = raw_labels[client_idx]
            logger.debug(f"Client dataset size: {client_data.shape[0]}")
            ret_datasets.append(ClientDataset(client_data, client_labels, transform=transform))
        return ret_datasets

    def generate_mini_test_data(self, proportion):
        X, y = self.test_data.data, self.test_data.targets
        mini_X, mini_y = [], []
        for each_class in range(len(self.test_data.classes)):
            indices = np.where(y == each_class)[0]
            class_num = int(np.round(len(indices) * proportion))
            # assert class_num > 0
            mini_loader_indices = indices[:class_num]
            mini_X.append(X[mini_loader_indices])
            mini_y.append(y[mini_loader_indices])
        mini_X = np.concatenate(mini_X, axis=0)
        mini_y = np.concatenate(mini_y, axis=0, dtype=np.int64)
        return mini_X, mini_y

    def get_personalized_test_data(self, distribution: np.ndarray)-> Tuple[Tensor, Tensor]:
        X, y = self.test_data.data, self.test_data.targets
        test_X, test_y = [], []
        distribution = (distribution * len(y) / 10).astype(int)
        for each_class in range(len(distribution)):
            indices = torch.where(y == each_class)[0]
            class_num = int(np.round(distribution[each_class]))
            test_loader_indices = indices[:class_num]
            test_X.append(X[test_loader_indices])
            test_y.append(y[test_loader_indices])
        test_X = torch.cat(test_X, dim=0)
        test_y = torch.cat(test_y, dim=0)
        return test_X, test_y

    def get_personalized_test_loader(self, distribution: np.ndarray, test_transform, collate_fn, args):
        test_X, test_y = self.get_personalized_test_data(distribution)
        return DataLoader(ClientDataset(test_X, test_y, transform=test_transform),
                          batch_size=args.test_batch_size,
                          shuffle=False,
                          num_workers=args.dataset.num_workers,
                          collate_fn=collate_fn,
                          # pin_memory=True
                          )

    def prepare_test_loaders(self, mini_test_proportion, test_batch_size,
                             test_num_workers=0,
                             test_transform=None,
                             collate_fn=None) \
            -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training
        :param collate_fn:
        :param test_transform:
        :param test_num_workers:
        :param test_batch_size:
        :param mini_test_proportion:
        :return: Tuple[DataLoader, DataLoader] mini_test_loader, full_test_loader
        """
        mini_X, mini_y = self.generate_mini_test_data(mini_test_proportion)
        full_X, full_y = self.test_data.data, self.test_data.targets
        mini_test_loader = DataLoader(
            ClientDataset(mini_X, mini_y, transform=test_transform),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=collate_fn,
            # pin_memory=True,
        )
        full_test_loader = DataLoader(
            ClientDataset(full_X, full_y, transform=test_transform),
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=test_num_workers,
            collate_fn=collate_fn,
            # pin_memory=True,
        )
        return mini_test_loader, full_test_loader
