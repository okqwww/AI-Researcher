"""
Main script to run training and testing for VQ-VAE with Rotation-Rescaling Transform.

Implements:
- Data loading (CIFAR-10)
- Model assembly
- Two epochs of training
- Evaluation (FID and reconstruction loss)
- Results logging
"""
import os
import json
import torch
import torch.optim as optim

from model.vqvae import VQVAE
from data_processing.cifar10 import get_cifar10_loaders
from training.train import train_one_epoch
from testing.test import evaluate

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Config
    config = {
        'batch_size': 128,
        'num_workers': 2,
        'codebook_size': 1024,
        'embedding_dim': 64,
        'beta': 0.5,
        'ema_decay': 0.99,
        'lr': 2e-4,
        'epochs': 3,
    }

    # Data
    train_loader, test_loader = get_cifar10_loaders(batch_size=config['batch_size'], num_workers=config['num_workers'])

    # Model
    model = VQVAE(img_channels=3,
                  codebook_size=config['codebook_size'],
                  beta=config['beta'],
                  embedding_dim=config['embedding_dim'],
                  ema_decay=config['ema_decay']).to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    # Training
    logs = {'train': [], 'test': None}
    for epoch in range(1, config['epochs'] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        train_stats = train_one_epoch(model, train_loader, optimizer, device)
        print("Train:", train_stats)
        logs['train'].append(train_stats)

    # Testing
    test_stats = evaluate(model, test_loader, device)
    print("Test:", test_stats)
    logs['test'] = test_stats

    os.makedirs('/workplace/project/results', exist_ok=True)
    with open('/workplace/project/results/logs.json', 'w') as f:
        json.dump(logs, f, indent=2)

if __name__ == '__main__':
    main()
