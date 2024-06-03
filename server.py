import torch
import os
from PIL import Image
from torch import Tensor
from utils.config import Config

import numpy as np
import logging

logger = logging.getLogger(__name__)


class DataIter:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.it = iter(dataloader)

    def __call__(self, *args, **kwargs):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.dataloader)
            return next(self.it)


class Server:
    datas: list[Tensor] | torch.Tensor | None = None
    labels: list[Tensor] | torch.Tensor | None = None
    file_sizes: torch.Tensor | None = None
    new_data: list[Tensor] | torch.Tensor = []
    new_label: list[Tensor] | torch.Tensor = []
    fetch_hist: list[int] = []

    def __init__(self,
                 logit_length: int,
                 args: Config,
                 # train_transform: torchvision.transforms.Compose
                 ):
        self.device = args.device
        self.ensemble_method = args.hnsw.ensemble_method
        self.logit_length = logit_length

    def save_to_knowledge_base(self, synthesized_imgs: Tensor, synthesized_labels: Tensor):
        logger.debug("Saving synthesized data to knowledge base")
        # for img, lb in zip(synthesized_imgs, synthesized_labels):
        #     self.new_data.append(img)
        #     self.new_label.append(lb)
        self.new_data.append(synthesized_imgs)
        self.new_label.append(synthesized_labels)

    def merge_new_data(self, iteration: int, save_dir: str, replace=False):
        assert len(self.new_data) > 0, "No new data to merge"
        # self.new_data = torch.stack(self.new_data, dim=0)
        # self.new_label = torch.stack(self.new_label, dim=0)
        self.new_data = torch.cat(self.new_data, dim=0)
        self.new_label = torch.cat(self.new_label, dim=0)
        if len(self.new_data.shape) > 3:
        # Save the new data
            for _dir in [save_dir, f"{save_dir}/{iteration}"]:
                if not os.path.exists(_dir):
                    os.mkdir(_dir)
            file_sizes = []
            new_images = self.new_data.cpu().numpy()
            labels = self.new_label.numpy().astype(int)
            if new_images.mean() <= 2:
                new_images *= 255
            new_images = new_images.astype(np.uint8)
            if new_images.shape[-1] > 3:
                new_images = new_images.transpose((0, 2, 3, 1))
            file_names = []
            for i, (img, label) in enumerate(zip(new_images, labels)):
                Image.fromarray(img).save(f"{save_dir}/{iteration}/{i}_{label}.png", optimize=True, quality=95)
                file_names.append(f"{i}_{label}.png")

            # Calculate the size of the new data
            full_save_dir = os.path.join(save_dir, str(iteration))
            for file_name in file_names:
                file_path = os.path.join(full_save_dir, file_name)
                file_sizes.append(os.path.getsize(file_path))
            file_sizes = torch.tensor(np.array(file_sizes, dtype=int))
        else:
            file_sizes = torch.zeros(self.new_data.shape[0])

        if self.datas is None or replace:
            self.datas = self.new_data
            self.labels = self.new_label
            if save_dir != "":
                self.file_sizes = file_sizes
            else:
                file_sizes = torch.tensor([0])
        else:
            self.datas = torch.cat([self.datas, self.new_data], dim=0)
            self.labels = torch.cat([self.labels, self.new_label], dim=0)
            if save_dir != "":
                self.file_sizes = torch.cat([self.file_sizes, file_sizes], dim=0)
            else:
                file_sizes = torch.tensor([0])
        self.new_data, self.new_label = [], []
        return torch.sum(file_sizes).item()

    def get_distilled_data(self, idx: np.ndarray | None = None):
        if idx is None:
            idx = np.arange(len(self.datas))
        self.fetch_hist.append(len(idx))
        return self.datas[idx], self.labels[idx], torch.sum(self.file_sizes[idx]).item()

    def get_distilled_data_by_distribution(self, dist, tau, total_num=-1):
        selected_idx = []
        distribution = dist * (1 - tau) + tau
        distribution /= distribution.sum()
        if total_num == -1:
            single_max_num = torch.bincount(self.labels).min().item()
            distribution = distribution / distribution.max()
            cls_num = (single_max_num * distribution).astype(int)
        else:
            cls_num = (total_num * distribution).astype(int)
        for cls, num in enumerate(cls_num):
            idx = torch.where(self.labels == cls)[0]
            idx = idx[torch.randperm(len(idx))]
            selected_idx.append(idx[:num])
        selected_idx = torch.cat(selected_idx)
        self.fetch_hist.append(len(selected_idx))
        return self.datas[selected_idx], self.labels[selected_idx], torch.sum(self.file_sizes[selected_idx]).item()
