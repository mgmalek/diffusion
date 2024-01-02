import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from torch.utils.data import DataLoader, IterableDataset
from torchvision.datasets import CIFAR10, MNIST

from utils import to_cuda_nonblocking

# NOTE: Using an IterableDataset is probably suboptimal since it means we're sampling with replacement


class MNISTDataset(IterableDataset):
    IMAGE_SIZE = (1, 32, 32)

    def __init__(self, train: bool):
        self.mnist_ds = MNIST("./datasets/mnist", train=True, download=True)

    def __len__(self):
        return len(self.mnist_ds)

    def __getitem__(self, index):
        pil_img, cls_idx = self.mnist_ds[index]
        img = np.array(pil_img, dtype=np.float32)
        img = torch.from_numpy(img)
        img = rearrange(img, "h w -> 1 1 h w")
        img = TF.resize(img, size=(32, 32), antialias=True)
        img = rearrange(img, "1 1 h w -> 1 h w")
        img = (img / 255) * 2 - 1  # rescale from (0, 255) to (-1, 1)
        cls_idx = cls_idx + 1
        return img, cls_idx

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self))
            yield self[idx]


class CIFAR10Dataset(IterableDataset):
    IMAGE_SIZE = (3, 32, 32)

    def __init__(self, train: bool):
        self.cifar_ds = CIFAR10("./datasets/cifar10", train=train, download=True)

    def __len__(self):
        return len(self.cifar_ds)

    def __getitem__(self, index):
        pil_img, cls_idx = self.cifar_ds[index]
        img = np.array(pil_img, dtype=np.float32)
        img = torch.from_numpy(img)
        img = rearrange(img, "h w c -> c h w")
        img = (img / 255) * 2 - 1  # rescale from (0, 255) to (-1, 1)
        if random.random() < 0.5:
            img = TF.hflip(img)
        cls_idx = cls_idx + 1
        return img, cls_idx

    def __iter__(self):
        while True:
            idx = np.random.randint(0, len(self))
            yield self[idx]


def iter_dl_with_prefetch(dl_iter, device: str) -> torch.Tensor:
    batch = next(dl_iter)
    batch = to_cuda_nonblocking(batch)

    while True:
        # asynchronously move the next batch onto GPU
        next_batch = next(dl_iter)
        next_batch = to_cuda_nonblocking(next_batch)

        yield batch

        batch = next_batch


def get_train_dataset(name: str):
    if name == "mnist":
        return MNISTDataset(train=True)
    elif name == "cifar10":
        return CIFAR10Dataset(train=True)
    else:
        raise ValueError(f"Invalid {name=}")


def get_image_size(name: str):
    if name == "mnist":
        return MNISTDataset.IMAGE_SIZE
    elif name == "cifar10":
        return CIFAR10Dataset.IMAGE_SIZE
    else:
        raise ValueError(f"Invalid {name=}")


def get_train_dataloader(dataset_name: str, batch_size: int, num_workers: int) -> DataLoader:
    train_ds = get_train_dataset(name=dataset_name)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
        pin_memory=True,
    )

    return train_dl
