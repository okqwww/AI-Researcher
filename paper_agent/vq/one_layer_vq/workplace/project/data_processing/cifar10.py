import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar10_dataloaders(root: str, batch_size: int = 128, num_workers: int = 2, image_size: int = 32) -> Tuple[DataLoader, DataLoader]:
    """
    Download (if needed) and create CIFAR-10 train/test dataloaders.

    - Inputs are normalized to [-1, 1].
    - Output images are 3 x image_size x image_size.
    """
    os.makedirs(root, exist_ok=True)
    tfm = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_set = datasets.CIFAR10(root=root, train=True, transform=tfm, download=True)
    test_set = datasets.CIFAR10(root=root, train=False, transform=tfm, download=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
