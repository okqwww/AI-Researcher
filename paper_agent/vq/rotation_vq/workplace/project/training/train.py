"""
Training loop for VQ-VAE with Rotation-Rescaling Transform quantizer.

Implements:
- Reconstruction loss (MSE)
- VQ commitment+codebook loss
- Logging of perplexity and cluster usage
- GPU support and OOM handling
"""
from typing import Dict
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler


def train_one_epoch(model, loader, optimizer, device: torch.device) -> Dict[str, float]:
    model.train()
    scaler = GradScaler(enabled=torch.cuda.is_available())
    total_rec = 0.0
    total_vq = 0.0
    total_loss = 0.0
    total_ppl = 0.0
    total_cuse = 0.0
    n_batches = 0
    for batch in loader:
        x, _ = batch
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        try:
            with autocast(enabled=torch.cuda.is_available()):
                x_rec, vq_loss, stats = model(x)
                rec_loss = F.mse_loss(x_rec, x)
                loss = rec_loss + vq_loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        total_rec += rec_loss.item()
        total_vq += vq_loss.item()
        total_loss += loss.item()
        total_ppl += stats[0].item()
        total_cuse += stats[1].item()
        n_batches += 1
    return {
        'rec_loss': total_rec / max(n_batches, 1),
        'vq_loss': total_vq / max(n_batches, 1),
        'loss': total_loss / max(n_batches, 1),
        'perplexity': total_ppl / max(n_batches, 1),
        'cluster_use': total_cuse / max(n_batches, 1),
    }
