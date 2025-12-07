import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from typing import Dict
import math


def _perplexity_from_usage(usage_vec: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Compute codebook perplexity from a probability vector over codes.
    perplexity = exp(-sum_k p_k log p_k)
    """
    p = usage_vec.clamp(min=eps)
    entropy = -(p * p.log()).sum().item()
    return math.exp(entropy)


def train_vqvae(model, train_loader, optimizer, device, epochs: int = 2, log_every: int = 100):
    scaler = GradScaler(enabled=torch.cuda.is_available())
    history = {"loss": [], "perplexity": [], "usage_top": []}
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        usage_accum = None
        n_batches = 0
        for i, (imgs, _) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=torch.cuda.is_available()):
                recons, qinfo = model(imgs)
                recon_loss = F.mse_loss(recons, imgs)
                loss = recon_loss + qinfo['commitment_loss'] + qinfo['codebook_loss']
            scaler.scale(loss).backward()
            # Clip to stabilize
            torch.nn.utils.clip_grad_norm_(list(model.encoder.parameters()) + list(model.decoder.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            # accumulate usage for perplexity
            usage = qinfo.get('usage', None)
            if usage is not None:
                usage_cpu = usage.detach().mean(dim=0) if usage.dim() > 1 else usage.detach()
                usage_accum = usage_cpu if usage_accum is None else usage_accum + usage_cpu
            n_batches += 1

            if (i + 1) % log_every == 0:
                print(f"Epoch {epoch} Iter {i+1}/{len(train_loader)} Loss {running / log_every:.4f}")
                running = 0.0
        # epoch aggregates
        avg_loss = running / max(1, (i + 1))
        history["loss"].append(avg_loss)
        if usage_accum is not None:
            usage_epoch = (usage_accum / max(1, n_batches))
            perplexity = _perplexity_from_usage(usage_epoch)
            # record top-10 used codes for quick inspection
            top_vals, top_idx = torch.topk(usage_epoch, k=min(10, usage_epoch.numel()))
            history["perplexity"].append(perplexity)
            history["usage_top"].append({"indices": top_idx.tolist(), "values": [float(v) for v in top_vals]})
        else:
            history["perplexity"].append(float('nan'))
            history["usage_top"].append({"indices": [], "values": []})
    return history
