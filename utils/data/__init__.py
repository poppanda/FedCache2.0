import os
from .cifar10 import CIFAR10Dataset
from .cinic10 import CINIC10Dataset
from .cifar100 import CIFAR100Dataset
from .basic_dataset import BasicDataset
from .urbansound import UrbanSoundDataset
from .tmd import TMDDataset
import logging
import numpy as np

# datasets: https://pytorch.org/vision/main/datasets.html

logger = logging.getLogger(__name__)


def load_data(dataset_name: str, save_dir: str = None, image_size: int = 224, extend_size: int = 64)\
        -> BasicDataset:
    """
    Load data from save_dir
    :param extend_size:
    :param dataset_name: the name of the dataset
    :param save_dir: directory where data is saved
    :param image_size: the target size of the image
    :return:
    """
    dataset_name = dataset_name.lower()
    save_dir = os.path.expanduser(save_dir)
    if dataset_name == "cifar10":
        return CIFAR10Dataset(save_dir, image_size, extend_size)
    elif dataset_name == "cinic10":
        return CINIC10Dataset(save_dir, image_size, extend_size)
    elif dataset_name =="cifar100":
        return CIFAR100Dataset(save_dir, image_size, extend_size)
    elif dataset_name == 'urbansound':
        return UrbanSoundDataset(save_dir)
    elif dataset_name == 'tmd':
        return TMDDataset(save_dir)
    else:
        raise ValueError("Dataset not implemented")


# def augment_image(rng, img):
#     rngs = random.split(rng, 8)
#     # Random left-right flip
#     img = dm_pix.random_flip_left_right(rngs[0], img)
#     # Color jitter
#     img_jt = img
#     img_jt = img_jt * random.uniform(rngs[1], shape=(1,), minval=0.5, maxval=1.5)  # Brightness
#     img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
#     img_jt = dm_pix.random_contrast(rngs[2], img_jt, lower=0.5, upper=1.5)
#     img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
#     img_jt = dm_pix.random_saturation(rngs[3], img_jt, lower=0.5, upper=1.5)
#     img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
#     img_jt = dm_pix.random_hue(rngs[4], img_jt, max_delta=0.1)
#     img_jt = jax.lax.clamp(0.0, img_jt, 1.0)
#     should_jt = random.bernoulli(rngs[5], p=0.8)
#     img = jnp.where(should_jt, img_jt, img)
#     # Random grayscale
#     should_gs = random.bernoulli(rngs[6], p=0.2)
#     img = jax.lax.cond(should_gs,  # Only apply grayscale if true
#                        lambda x: dm_pix.rgb_to_grayscale(x, keep_dims=True),
#                        lambda x: x,
#                        img)
#     # Gaussian blur
#     sigma = random.uniform(rngs[7], shape=(1,), minval=0.1, maxval=2.0)
#     img = dm_pix.gaussian_blur(img, sigma=sigma[0], kernel_size=9)
#     # Normalization
#     img = img * 2.0 - 1.0
#     return img


def swap_proto_data(clients, swap_times):
    swap_idx = np.random.randint(0, len(clients), (swap_times, 2))
    for i, j in swap_idx:
        clients[i].x_proto, clients[j].x_proto = clients[j].x_proto.clone().detach(), clients[i].x_proto.clone().detach()
        clients[i].y_proto, clients[j].y_proto = clients[j].y_proto.clone().detach(), clients[i].y_proto.clone().detach()

