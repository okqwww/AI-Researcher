"""
Testing procedures for VQ-VAE.

Computes reconstruction metrics (MSE, PSNR, SSIM) and FID on CIFAR-10 test set.
"""
from typing import Dict
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .fid import get_inception_features, calculate_fid

def psnr_from_mse(mse: torch.Tensor, max_val: float = 1.0) -> float:
    # PSNR in dB
    import math
    if mse <= 0:
        return float('inf')
    return 10.0 * math.log10((max_val ** 2) / float(mse))

def ssim_simple(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2) -> float:
    """
    A simple SSIM approximation computed per-image over the whole frame (no sliding window).
    Inputs expected in [0,1]. Returns average SSIM over batch.
    """
    # Flatten spatial dims
    B = x.size(0)
    x_flat = x.view(B, x.size(1), -1)
    y_flat = y.view(B, y.size(1), -1)
    mu_x = x_flat.mean(-1)
    mu_y = y_flat.mean(-1)
    var_x = x_flat.var(-1, unbiased=False)
    var_y = y_flat.var(-1, unbiased=False)
    cov_xy = ((x_flat - mu_x.unsqueeze(-1)) * (y_flat - mu_y.unsqueeze(-1))).mean(-1)
    ssim_c = ((2 * mu_x * mu_y + C1) * (2 * cov_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))
    # Average over channels then batch
    return float(ssim_c.mean().item())

def evaluate(model, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    total_rec = 0.0
    n_batches = 0
    orig_images = []
    rec_images = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x_rec, vq_loss, stats = model(x)
            rec_loss = F.mse_loss(x_rec, x)
            total_rec += rec_loss.item()
            n_batches += 1
            orig_images.append(x.detach().cpu())
            rec_images.append(x_rec.detach().cpu())
    # Stack and compute FID
    orig = torch.cat(orig_images, dim=0)
    rec = torch.cat(rec_images, dim=0)
    # Build loaders for feature extraction
    from torch.utils.data import TensorDataset
    orig_loader = torch.utils.data.DataLoader(TensorDataset(orig, torch.zeros(orig.size(0))), batch_size=64)
    rec_loader = torch.utils.data.DataLoader(TensorDataset(rec, torch.zeros(rec.size(0))), batch_size=64)
    feats_orig = get_inception_features(orig_loader, device)
    feats_rec = get_inception_features(rec_loader, device)
    fid = calculate_fid(feats_orig, feats_rec)
    # PSNR/SSIM
    avg_mse = total_rec / max(n_batches, 1)
    psnr = psnr_from_mse(avg_mse)
    ssim = ssim_simple(orig, rec)
    return {
        'rec_loss': avg_mse,
        'psnr': psnr,
        'ssim': ssim,
        'fid': fid,
    }
