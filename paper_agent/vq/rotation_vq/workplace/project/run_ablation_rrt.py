"""
Run ablation comparing RRT angle-preserving gradient transport vs standard straight-through.
Trains each setting for 2 epochs on CIFAR-10 and evaluates MSE, PSNR, SSIM, FID.
Logs results to /workplace/project/results/ablation_rrt.json.
"""
import os
import json
import torch
import torch.optim as optim

from model.vqvae import VQVAE
from data_processing.cifar10 import get_cifar10_loaders
from training.train import train_one_epoch
from testing.test import evaluate


def run_ablation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    os.makedirs('/workplace/project/results', exist_ok=True)

    config = {
        'batch_size': 128,
        'num_workers': 2,
        'codebook_size': 1024,
        'embedding_dim': 64,
        'lr': 2e-4,
        'epochs': 2,
        'ema_decay': 0.99,
        'beta': 0.5,
    }

    train_loader, test_loader = get_cifar10_loaders(batch_size=config['batch_size'], num_workers=config['num_workers'])

    for use_rrt in [True, False]:
        tag = 'rrt' if use_rrt else 'straight_through'
        model = VQVAE(img_channels=3,
                      codebook_size=config['codebook_size'],
                      beta=config['beta'],
                      embedding_dim=config['embedding_dim'],
                      ema_decay=config['ema_decay'],
                      use_rrt=use_rrt).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])

        train_logs = []
        for epoch in range(1, config['epochs'] + 1):
            stats = train_one_epoch(model, train_loader, optimizer, device)
            train_logs.append(stats)
        test_stats = evaluate(model, test_loader, device)
        results[tag] = {
            'train': train_logs,
            'test': test_stats,
        }

    with open('/workplace/project/results/ablation_rrt.json', 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == '__main__':
    run_ablation()
