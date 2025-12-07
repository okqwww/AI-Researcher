"""
CIFAR-10 data processing pipeline.

Downloads the dataset to /workplace/project/data if not present, applies standard
transforms, and provides DataLoaders for training and testing.

FID evaluation resizes images to 299x299 for InceptionV3 features.
"""
import os
from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T

DATA_DIR = "/workplace/project/data"


def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    os.makedirs(DATA_DIR, exist_ok=True)
    # Transforms for VQ-VAE: [0,1]
    transform = T.Compose([
        T.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader
