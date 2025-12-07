"""
Utility functions and autograd helpers for the VQ-VAE with Rotation-Rescaling Transform (RRT).

This file integrates ideas adapted from multiple references:
- Neural Discrete Representation Learning (van den Oord et al., 2017)
- Categorical Reparameterization with Gumbel-Softmax (Jang et al., 2017)
- Reference implementations: airalcorn2-vqvae-pytorch, CompVis latent-diffusion

All code is rewritten to fit a self-contained architecture and documented with origins and modifications.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_softmax_sample(logits: torch.Tensor, temperature: float, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
    Sample from Gumbel-Softmax distribution.

    Origin: Adapted from EdoardoBotta-Gaussian-Mixture-VAE (reimplemented).
    Paper: Jang et al. "Categorical Reparameterization with Gumbel-Softmax".

    Args:
        logits: [..., K] unnormalized log-probabilities for K categories.
        temperature: Softmax temperature.
        device: device for sampling noise.

    Returns:
        Softmax probabilities of shape [..., K].
    """
    gumbel_noise = -torch.log(-torch.log(torch.randn_like(logits).to(device) + 1e-20) + 1e-20)
    y = logits + gumbel_noise
    return F.softmax(y / temperature, dim=-1)


def measure_perplexity(predicted_indices: torch.Tensor, n_embed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluate cluster perplexity and usage.

    Origin: Adapted from a public implementation note by Andrej Karpathy (reimplemented).
    Reference repo mentioned in survey notes.

    Args:
        predicted_indices: Long tensor of shape [N] containing selected code indices.
        n_embed: Number of embeddings in the codebook.

    Returns:
        (perplexity, cluster_use):
         - perplexity: measures how uniformly codes are used. Max equals n_embed.
         - cluster_use: number of clusters with non-zero usage in the batch.
    """
    encodings = F.one_hot(predicted_indices, n_embed).float().reshape(-1, n_embed)
    avg_probs = encodings.mean(0)
    perplexity = (-(avg_probs * torch.log(avg_probs + 1e-10)).sum()).exp()
    cluster_use = torch.sum(avg_probs > 0)
    return perplexity, cluster_use


def householder_apply(v: torch.Tensor, w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Apply the Householder transformation that maps unit vector v to unit vector w, to vector x.

    We use the classic construction:
      u = (v - w) / ||v - w||, H = I - 2 u u^T, and H v = w.
    This avoids explicit construction of H and applies it via vector operations.

    Args:
        v: [..., D] unit vectors (encoder direction).
        w: [..., D] unit vectors (codebook direction).
        x: [..., D] vector to transform.

    Returns:
        H x where H maps v -> w.
    """
    # If v == w (or nearly), H becomes identity. We guard against numerical issues.
    u = v - w
    # Norm with small epsilon to avoid division by zero.
    u_norm = torch.norm(u, dim=-1, keepdim=True).clamp_min(1e-8)
    u = u / u_norm
    # Householder reflection: x - 2 u (u^T x)
    ux = torch.sum(u * x, dim=-1, keepdim=True)
    hx = x - 2.0 * u * ux
    return hx


class RRTStraightThrough(torch.autograd.Function):
    """
    Custom autograd to transport gradients through the quantization layer using
    Rotation and Rescaling Transform (RRT) while keeping the forward output equal
    to the nearest codebook vector.

    Forward:
        z_e -> e_q (nearest code vector) treated as quantized output to decoder.
    Backward:
        Let R be Householder reflection mapping z_dir to e_dir, and s = ||e||/||z||.
        We transport the gradient g via s * R^T g = s * R g (R is symmetric), which
        preserves angle with the codebook direction and provides meaningful signals
        to the encoder.

    Important: The rotation and rescaling are treated as constants w.r.t gradients,
    i.e., we do not backpropagate into the codebook by this path (EMA handles updates).
    """

    @staticmethod
    def forward(ctx, z_e: torch.Tensor, e_q: torch.Tensor) -> torch.Tensor:
        # Compute per-vector unit directions and rescaling factor s.
        # Shapes: [..., D]
        z_norm = torch.norm(z_e, dim=-1, keepdim=True).clamp_min(1e-8)
        e_norm = torch.norm(e_q, dim=-1, keepdim=True).clamp_min(1e-8)
        z_dir = z_e / z_norm
        e_dir = e_q / e_norm
        s = (e_norm / z_norm).detach()  # stop gradient through s

        # Save for backward (as detached constants)
        ctx.save_for_backward(z_dir.detach(), e_dir.detach(), s)
        # Forward output equals quantized vector e_q.
        return e_q

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        z_dir, e_dir, s = ctx.saved_tensors
        # Apply Householder reflection that maps z_dir -> e_dir to grad_output.
        # Preserve angle with codebook direction by transporting via R.
        # R is symmetric, so R^T = R.
        transformed = householder_apply(z_dir, e_dir, grad_output)
        grad_z = s * transformed
        # No gradient flows to e_q (EMA updates codebook). We return grad for z_e only.
        return grad_z, None
