"""
Vector Quantizer with Exponential Moving Average (EMA) updates and
Rotation-Rescaling Transform (RRT) straight-through gradient routing.

Origins & References:
- Neural Discrete Representation Learning (van den Oord et al., 2017)
- airalcorn2-vqvae-pytorch (reimplemented logic for EMA updates and losses)
- This project integrates a novel gradient transport based on Householder rotation
  and rescaling, as defined in utils.RRTStraightThrough.

All code is rewritten and self-contained; no external imports from reference repos.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import RRTStraightThrough, measure_perplexity

class VectorQuantizerEMA(nn.Module):
    """
    VQ module implementing codebook with EMA updates and commitment loss.

    Forward returns the quantized vectors e_q, but we register a custom backward
    via RRTStraightThrough to transport gradients through quantization preserving
    angle with codebook vector, if use_rrt=True. Otherwise, uses standard straight-through.
    """

    def __init__(self, embedding_dim: int, num_embeddings: int,
                 decay: float = 0.99, epsilon: float = 1e-5,
                 beta: float = 0.25, use_rrt: bool = True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.decay = decay
        self.epsilon = epsilon
        self.beta = beta
        self.use_rrt = use_rrt

        # Codebook embeddings (dictionary) e_i in R^{K x D}
        limit = 3 ** 0.5
        embed = torch.empty(num_embeddings, embedding_dim, dtype=torch.float32).uniform_(-limit, limit)
        self.register_buffer('embedding', embed)
        # EMA buffers following Sonnet's implementation (counts and sums)
        self.register_buffer('cluster_size', torch.zeros(num_embeddings, dtype=torch.float32))
        self.register_buffer('embed_avg', embed.clone())

    @torch.no_grad()
    def _ema_update(self, z: torch.Tensor, encoding_indices: torch.Tensor):
        """
        Sonnet-style EMA updates with Laplace smoothing and dead-code reinitialization.
        z: [N, D] on same device as embedding
        encoding_indices: [N]
        """
        device = z.device
        # Cast z to codebook dtype (float32) to avoid AMP dtype mismatches
        z = z.to(self.embed_avg.dtype)
        K = self.num_embeddings
        # One-hot encodings [N, K]
        encodings = F.one_hot(encoding_indices, K).to(self.embed_avg.dtype)  # [N, K]
        # Current batch stats
        cluster_size_batch = encodings.sum(0)  # [K]
        embed_sum_batch = encodings.t() @ z  # [K, D]

        # Exponential moving average update
        self.cluster_size.mul_(self.decay).add_(cluster_size_batch * (1.0 - self.decay))
        self.embed_avg.mul_(self.decay).add_(embed_sum_batch * (1.0 - self.decay))

        # Laplace smoothing of counts to avoid zeros
        n = self.cluster_size.sum()
        cluster_size = (self.cluster_size + self.epsilon)
        # Renormalize counts to sum to n (as in Sonnet)
        cluster_size = cluster_size / cluster_size.sum().clamp_min(1e-8) * n

        # Normalized embeddings
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

        # Dead code handling: reinitialize rarely used codes from random z
        dead_codes = (cluster_size < 1.0)
        if torch.any(dead_codes) and z.shape[0] > 0:
            num_dead = int(dead_codes.sum().item())
            rand_idx = torch.randint(0, z.shape[0], (num_dead,), device=device)
            self.embed_avg[dead_codes] = z[rand_idx]
            cluster_size[dead_codes] = 1.0
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)

        # In-place update to preserve buffer identity
        self.embedding.copy_(embed_normalized)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z_e: [B, D, H, W] or [N, D] encoder outputs.
        Returns:
            e_q: quantized vectors same shape as z_e.
            indices: selected code indices.
            vq_loss: commitment + codebook loss for training stability.
            stats: perplexity and cluster_use for logging
        """
        # Flatten spatial dims if present
        orig_shape = z_e.shape
        if z_e.dim() == 4:
            B, D, H, W = orig_shape
            z = z_e.permute(0, 2, 3, 1).contiguous().view(-1, D)
        elif z_e.dim() == 2:
            z = z_e
            D = z.shape[-1]
            B = None
        else:
            raise ValueError(f"Unsupported shape for z_e: {orig_shape}")

        # Compute distances to codebook
        embed = self.embedding  # [K, D], float32 buffer
        # Cast z to embedding dtype for distance computation stability
        z32 = z.to(embed.dtype)
        z_sq = torch.sum(z32 ** 2, dim=1, keepdim=True)  # [N, 1]
        e_sq = torch.sum(embed ** 2, dim=1)  # [K]
        ze = z32 @ embed.t()  # [N, K]
        distances = z_sq + e_sq.unsqueeze(0) - 2 * ze
        # Choose nearest embedding
        indices = torch.argmin(distances, dim=1)
        e_q_flat = F.embedding(indices, embed)  # [N, D] float32

        # Update codebook via EMA (stop-grad)
        self._ema_update(z.detach(), indices.detach())

        # Losses (treat decoder input as constants for encoder path)
        # Compute losses in float32 for stability
        codebook_loss = F.mse_loss(e_q_flat, z32.detach())
        commit_loss = self.beta * F.mse_loss(z32, e_q_flat.detach())
        vq_loss = codebook_loss + commit_loss

        # Straight-through gradient handling
        if self.use_rrt:
            # RRT; forward remains e_q, backward transports angle-preserving grad
            e_q_st = e_q_flat.to(z.dtype)
            e_q_st = RRTStraightThrough.apply(z, e_q_st)
        else:
            # Standard straight-through estimator: treat quantization as identity in backward
            # e_q = z + (e_q_flat - z).detach()
            e_q_st = z + (e_q_flat.to(z.dtype) - z).detach()

        # Reshape back
        if z_e.dim() == 4:
            e_q = e_q_st.view(B, H, W, D).permute(0, 3, 1, 2).contiguous()
        else:
            e_q = e_q_st

        # Stats
        perplexity, cluster_use = measure_perplexity(indices, self.num_embeddings)
        stats = torch.stack([perplexity, cluster_use.float()])
        return e_q, indices, vq_loss, stats
