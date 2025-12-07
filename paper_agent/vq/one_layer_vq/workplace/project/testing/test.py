import os
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from typing import Dict
from .fid import FIDCalculator


def evaluate_reconstructions(model, test_loader: DataLoader, device, stats_path: str) -> Dict[str, float]:
    model.eval()
    recon_loss_total = 0.0
    n = 0
    fid_calc = FIDCalculator(stats_path=stats_path, device=device)

    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'results', 'samples'), exist_ok=True)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(test_loader):
            imgs = imgs.to(device, non_blocking=True)
            recons, _ = model(imgs)
            recon_loss = F.mse_loss(recons, imgs, reduction='sum').item()
            recon_loss_total += recon_loss
            n += imgs.numel()
            # collect for FID
            fid_calc.update(recons)
            if i == 0:
                # save a grid of first batch reconstructions
                grid = torch.cat([imgs[:8], recons[:8]], dim=0)
                grid = (grid.clamp(-1, 1) + 1) / 2
                save_image(grid, os.path.join(os.path.dirname(__file__), '..', 'results', 'samples', 'recon_grid.png'), nrow=8)

    fid = fid_calc.compute()
    return {"mse": recon_loss_total / n, "fid": float(fid)}
