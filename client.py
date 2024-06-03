import os
import torch
import random
import logging
import numpy as np
from torch import nn
from torch import Tensor
from typing import Dict, List
from torch.utils.data import DataLoader
from utils.data.client_dataset import ClientDataset

logger = logging.getLogger(__name__)


class Client:
    transmit_size = 0
    eval_acc: float = 0.0
    personalized_eval_acc: float = 0.0
    proto_optimizer: torch.optim.Optimizer | None = None
    downloaded_idx = None
    x_proto, y_proto = None, None
    personalized_test_loader = None
    feats, feat_labels = None, None
    feat_loader = None
    model_pool = []

    def __init__(self,
                 client_id: int,
                 model_type: str,
                 model: nn.Module,
                 dataset: ClientDataset,
                 max_sync_data_num: int,
                 output_dim: int = 10,
                 dataset_args: Dict | None = None,
                 optimizer_args=None,
                 num_prototypes_per_class: int = 10,
                 criterion: str = "cross_entropy",
                 device='cuda',
                 ):
        """
        Create a client
        :param client_id: identity of the client (used for logging)
        :param model_type: specify the model type (e.g. resnet18, use Utils.Model.create_model to create the model)
        :param output_dim:
        """
        if optimizer_args is None:
            optimizer_args = {}
        self.client_id = client_id
        self.output_dim = output_dim
        self.model_type = model_type
        self.device = torch.device(device)
        # self.trainer = create_model(model_type, output_dim=output_dim,
        #                             optimizer=args.optimizer,
        #                             lr=args.lr,
        #                             full_img_size=args.dataset.full_img_size)
        model.to(self.device)
        self.model = model
        optimziers = {
            "adam": torch.optim.Adam,
            "sgd": torch.optim.SGD
        }
        optimizer = optimizer_args.pop('name', 'adam')
        self.optimizer = optimziers[optimizer](model.parameters(),
                                               **optimizer_args)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.4)
        criterions = {
            "cross_entropy": nn.CrossEntropyLoss,
            "mse": nn.MSELoss
        }
        self.criterion = criterions[criterion]()
        self.dataset: ClientDataset = dataset
        # print(self.dataset.data)
        self.batch_size = dataset_args["batch_size"]
        self.num_workers = dataset_args["num_workers"]
        self.data_loader = DataLoader(dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True)
        self.num_prototypes_per_class = num_prototypes_per_class
        self.max_sync_data_num = max_sync_data_num

    def train(self, save_feat=False, step_lr=False):
        logger.debug(f"Client {self.client_id} training...")
        # self.model.to(self.device)
        self.model.train()
        if save_feat:
            new_feats_list: List[Tensor] = []
            new_labels_list: List[Tensor] = []
        for imgs, lbs in self.data_loader:
            imgs, lbs = imgs.to(self.device, non_blocking=True), lbs.to(self.device, non_blocking=True)
            logits, feats = self.model(imgs, return_feats=True)
            if save_feat:
                new_feats_list.append(feats.clone().detach())
                new_labels_list.append(lbs.clone().detach())
            loss = self.criterion(logits, lbs)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        if step_lr:
            self.lr_scheduler.step()
        if save_feat:
            new_feats = torch.cat(new_feats_list, dim=0).cpu()
            new_labels = torch.cat(new_labels_list, dim=0).cpu()
            self.feats = new_feats if self.feats is None \
                else torch.cat([self.feats, new_feats], dim=0)
            self.feat_labels = new_labels if self.feat_labels is None \
                else torch.cat([self.feat_labels, new_labels], dim=0)

    def eval(self, dataloader: DataLoader):
        self.model.eval()
        # self.model.to(self.device)
        correct_class, count = 0, 0
        for imgs, lbs in dataloader:
            imgs, lbs = imgs.to(self.device), lbs.to(self.device)
            logits = self.model(imgs)
            correct_class += (logits.argmax(dim=1) == lbs).sum().item()
            count += imgs.shape[0]
        eval_acc = correct_class / count
        # self.model.to(torch.device("cpu"))
        return eval_acc

    def init_synthesized_data(self, resume=False, **args):
        if resume:
            x_proto, y_proto = self._load_proto(**args)
        else:
            x_proto, y_proto = self._init_proto()
        self.x_proto = x_proto.to(self.device)
        self.y_proto = y_proto.float().to(self.device)

    def merge_feat(self):
        self.feat_loader = DataLoader(ClientDataset(self.feats, self.feat_labels),
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      pin_memory=True)

    def clear_feat(self):
        self.feats, self.feat_labels = None, None
        self.feat_loader = None

    def _init_proto(self):
        data, labels = self.dataset.data, self.dataset.labels
        x_proto, y_proto = [], []
        for cls in range(self.output_dim):
            cls_idx = labels == cls
            cls_data = data[cls_idx]
            if isinstance(data, np.ndarray):
                selected_idx = np.random.permutation(np.sum(cls_idx))[:self.num_prototypes_per_class]
            elif isinstance(data, Tensor):
                selected_idx = torch.randperm(torch.sum(cls_idx))[:self.num_prototypes_per_class]
            else:
                raise ValueError("Data type not supported")
            x_proto.append(cls_data[selected_idx])
            y_proto.append(np.full(selected_idx.shape[0], cls))
        x_proto = np.concatenate(x_proto, dtype=np.float32)
        x_proto = torch.tensor(x_proto)
        y_proto = torch.tensor(np.concatenate(y_proto))
        y_proto = torch.nn.functional.one_hot(y_proto, self.output_dim)
        # print("Max val", x_proto.max())
        return x_proto, y_proto

    def save_proto(self, save_dir, suffix):
        x_proto, y_proto = self.get_synthesized_data()
        torch.save(x_proto, os.path.join(save_dir, f"X_{self.client_id}-{suffix}.pt"))
        torch.save(y_proto, os.path.join(save_dir, f"Y_{self.client_id}-{suffix}.pt"))

    def _load_proto(self, save_dir, suffix):
        x_proto = torch.load(os.path.join(save_dir, f"X_{self.client_id}-{suffix}.pt"))
        y_proto = torch.load(os.path.join(save_dir, f"Y_{self.client_id}-{suffix}.pt"))
        return x_proto, y_proto

    @staticmethod
    def lb_margin_th(logits):
        dim = logits.shape[-1]
        val, idx = torch.topk(logits, k=2)
        # margin = torch.minimum((val[:, 0] - val[:, 1]), 1 / dim)
        margin = torch.clamp(val[:, 0] - val[:, 1], min=0, max=1 / dim)
        return -margin

    @staticmethod
    def mean_squared_loss(logits, labels):
        if len(logits.shape) != len(labels.shape):
            labels = torch.nn.functional.one_hot(labels, logits.shape[-1])
        return torch.sum((logits - labels) ** 2 * 0.5, dim=-1)

    def nfr(self, x_target_feat, x_proto, y_proto, reg=1e-6):
        x_proto_feat = self.model(x_proto, return_feats=True)[1]
        k_pp = x_proto_feat @ x_proto_feat.T
        k_tp = x_target_feat @ x_proto_feat.T
        k_pp_reg = (k_pp + abs(reg) * torch.trace(k_pp) * torch.eye(k_pp.shape[0], device=self.device) / k_pp.shape[0])
        # print(k_pp_reg.shape, y_proto.shape, k_tp.shape)
        try:
            pred = k_tp @ torch.linalg.solve(k_pp_reg, y_proto)
        except:
            pred = k_tp @ torch.linalg.solve(k_pp_reg + 1e-3 * torch.eye(k_pp_reg.shape[0], device=self.device), y_proto)
        return pred

    # @torch.compile(dynamic=True)
    def train_proto_step(self, feats, lbs, use_flip=False):
        if use_flip:
            print("WARNING: Use flip, check the optimizer settings")
            x_proto_flip = torch.flip(self.x_proto, dims=(-2,))
            x_proto = torch.cat([self.x_proto, x_proto_flip], dim=0)
            y_proto = torch.cat([self.y_proto, self.y_proto], dim=0)
        # print(self.x_proto)
        lb_loss = self.lb_margin_th(self.y_proto).mean()
        predictions = self.nfr(feats, self.x_proto, self.y_proto)
        kernel_loss = self.mean_squared_loss(predictions, lbs).mean()
        loss = kernel_loss + lb_loss
        self.proto_optimizer.zero_grad()
        self.model.zero_grad()
        loss.backward()
        self.proto_optimizer.step()
        # print("Loss: ", loss.item(), "Kernel loss: ", kernel_loss.item(), "LB loss: ", lb_loss.item())

    def init_proto_optimizer(self, optimize_label=False):
        if len(self.x_proto.shape) > 3:
            self.x_proto = torch.clamp(self.x_proto.clone().detach(), 0, 1)
        else:
            self.x_proto = self.x_proto.clone().detach()
        self.y_proto = self.y_proto.clone().detach()
        self.x_proto.requires_grad_(True)
        if optimize_label:
            self.y_proto.requires_grad_(True)
            self.proto_optimizer = torch.optim.Adam([self.x_proto, self.y_proto], lr=3e-4)
        else:
            self.proto_optimizer = torch.optim.Adam([self.x_proto], lr=3e-4)

    def modify_synthesized_data(self):
        self.model.train()
        self.model.to(self.device)
        self.model.zero_grad()
        for feats, labels in self.feat_loader:
            feats, labels = feats.to(self.device), labels.to(self.device)
            self.train_proto_step(feats, labels)
        # self.model.to(torch.device("cpu"))

    def get_synthesized_data(self):
        x_proto = self.x_proto.clone().detach()
        y_proto = self.y_proto.clone().detach()
        return x_proto, y_proto

    def get_un_stored_data_idx(self, new_idx):
        if self.downloaded_idx is None:
            return new_idx
        else:
            return np.setdiff1d(new_idx, self.downloaded_idx, assume_unique=True)

    def save_data_idx(self, idx):
        if self.downloaded_idx is None:
            self.downloaded_idx = idx
        else:
            self.downloaded_idx = np.unique(np.concatenate([self.downloaded_idx, idx]))

    def save_synthesized_data(self, X, y, replace=False):
        if replace:
            self.dataset.clear_synthesized_data()
        self.dataset.add_synthesized_data(X, y)
        self.dataset.bound_synthesized_data_num(self.max_sync_data_num)
        self.data_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      shuffle=True,
                                      num_workers=self.num_workers,
                                      # pin_memory=True
                                      )

    # def test(self, dataloader: DataLoader):
    #     return self.trainer.eval_model(dataloader)

    # def generate_logits(self, switch_device: bool = True):
    #     self.trainer.eval()
    #     if switch_device:
    #         self.trainer.to(self.device)
    #     rec_logits = []
    #     rec_keys = []
    #     self.logits = {}
    #     with torchmodel.no_grad():
    #         for (keys, X, y) in self.data_loader:
    #             X, y = X.to(self.device), y.to(self.device)
    #             y_hat = self.trainer(X)
    #             correct_idx: torchmodel.Tensor = (y_hat.argmax(dim=1) == y)
    #             rec_logits_temp = y_hat[correct_idx]
    #             rec_keys_temp = keys[correct_idx.cpu()]
    #             rec_logits.append(rec_logits_temp.detach().cpu().numpy())
    #             rec_keys.append(rec_keys_temp.numpy())
    #     logger.debug(f"Client {self.client_id} generated {len(rec_keys)} logits.")
    #     rec_keys = np.concatenate(rec_keys, axis=0)
    #     rec_logits = np.concatenate(rec_logits, axis=0)
    #     logger.debug(f"keys shape: {rec_keys.shape} logits shape: {rec_logits.shape}")
    #     for key, logit in zip(rec_keys, rec_logits):
    #         self.logits[key] = logit
    #     if switch_device:
    #         self.train_state.to(torchmodel.device("cpu"))
    #
    # def get_logits(self):
    #     if self.logits is None:
    #         raise ValueError("Logits not recorded.")
    #     return self.logits

    def save_to_model_pool(self):
        self.model_pool.append(self.model.state_dict())

    def select_from_model_pool(self):
        idx = random.randint(0, len(self.model_pool) - 1)
        self.model.load_state_dict(self.model_pool[idx])

    def save_model(self, save_dir, prefix: str = "", step=0):
        torch.save(self.model.state_dict(), os.path.join(save_dir, f"{prefix}-model-{step}.pt"))

    def load_model(self, save_dir: str, prefix: str = "", step=0):
        self.model.load_state_dict(torch.load(os.path.join(save_dir, f"{prefix}-model-{step}.pt")))
