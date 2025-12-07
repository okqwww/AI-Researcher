import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

from .layers import Encoder, Decoder

class EMACodebook(nn.Module):
    """
    Exponential Moving Average (EMA) codebook updates adapted from Sonnet/DeepMind VQ-VAE v2.
    Origin: Based on the EMA update described in Neural Discrete Representation Learning (VQ-VAE)
    and common PyTorch ports (e.g., vqvae-pytorch, VQGAN-pytorch). Rewritten to be self-contained.
    """
    def __init__(self, codebook_size: int, embedding_dim: int, decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.decay = decay
        self.eps = eps

        self.embedding = nn.Embedding(codebook_size, embedding_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA buffers
        self.register_buffer('cluster_size', torch.zeros(codebook_size))
        self.register_buffer('embed_avg', torch.zeros(codebook_size, embedding_dim))

    @torch.no_grad()
    def ema_update(self, encodings: torch.Tensor, flat_inputs: torch.Tensor):
        """
        Update EMA statistics and embeddings.
        encodings: one-hot assignments [N, K]
        flat_inputs: inputs [N, D]
        """
        # Update cluster size and embed averages
        new_cluster_size = encodings.sum(0)
        self.cluster_size.mul_(self.decay).add_(new_cluster_size, alpha=1 - self.decay)

        embed_sum = encodings.transpose(0, 1) @ flat_inputs  # [K, D]
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Laplace smoothing
        n = self.cluster_size.sum()
        cluster_size = ((self.cluster_size + self.eps) / (n + self.codebook_size * self.eps)) * n

        # Normalize to get the new embedding weights
        embed_normalized = self.embed_avg / cluster_size.unsqueeze(1)
        self.embedding.weight.data.copy_(embed_normalized)

    def forward(self):
        return self.embedding.weight


def householder_vectors(z: torch.Tensor, e: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Householder reflection vectors u for mapping z -> e.
    H = I - 2 uu^T where u = (z - e) / ||z - e||.
    We return u; In backward, we will use H implicitly.
    """
    diff = z - e
    norm = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=eps)
    u = diff / norm
    return u.detach()  # treat as constant w.r.t gradient as per stop-gradient

class RotationRescaleQuantizeFn(torch.autograd.Function):
    """
    Custom autograd function performing vector quantization with rotation-rescaling gradient mapping.

    Forward: returns quantized codebook vectors (nearest neighbors).
    Backward: maps gradient through Householder reflection (orthogonal) and optional scaling, preserving
              angles between gradient and codebook vectors via orthogonal transform.
    """
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, codebook: torch.Tensor, indices: torch.Tensor):
        # inputs: [N, D], codebook: [K, D], indices: [N]
        e = codebook[indices]  # [N, D]
        # Compute scaling s so that s*H*z = e. For Householder H mapping z->e, s usually ~ 1 if norms equal,
        # but to guarantee match, set s = ||e|| / ||z||.
        s = (e.norm(dim=-1) / inputs.norm(dim=-1).clamp(min=1e-8)).unsqueeze(-1).detach()
        # Compute Householder reflection vectors u for z->e
        u = householder_vectors(inputs, e)  # [N, D]
        ctx.save_for_backward(u, s)
        ctx.codebook = codebook
        ctx.indices = indices
        # Forward pass output equals quantized e (decoder input)
        return e

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output: [N, D]
        u, s = ctx.saved_tensors
        # Implicitly apply H^T to grad_output; H is symmetric, H^T = H.
        # H g = (I - 2 u u^T) g = g - 2 u (u^T g)
        proj = (u * grad_output).sum(dim=-1, keepdim=True)
        grad_inputs = grad_output - 2.0 * u * proj
        grad_inputs = grad_inputs * s  # rescale, doesn't change angles.
        # No gradients to codebook or indices through this path (EMA updates handle codebook)
        return grad_inputs, None, None

class VectorQuantizerRR(nn.Module):
    """
    Vector quantization layer with rotation and rescaling gradient mapping and EMA codebook.

    - Finds nearest neighbor codebook entries for each encoder output vector.
    - Forward pass outputs quantized code vectors to decoder.
    - Backward pass uses custom rotation-rescale to propagate gradients.
    - EMA updates maintain codebook and mitigate collapse.
    """
    def __init__(self, codebook_size: int, embedding_dim: int, beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.codebook = EMACodebook(codebook_size, embedding_dim, decay=ema_decay)
        self.beta = beta
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # z_e: [B, D, H, W]
        B, D, H, W = z_e.shape
        flat_inputs = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # [N, D]
        codebook = self.codebook().detach()  # treat as constant for gradient through transform

        # Compute nearest codebook indices using L2 distance
        # dist = ||x||^2 + ||e||^2 - 2 x.e
        x_sq = (flat_inputs ** 2).sum(dim=1, keepdim=True)  # [N, 1]
        e_sq = (codebook ** 2).sum(dim=1).unsqueeze(0)  # [1, K]
        cross = flat_inputs @ codebook.t()  # [N, K]
        dist = x_sq + e_sq - 2 * cross
        indices = dist.argmin(dim=1)  # [N]

        # One-hot encodings for EMA update
        encodings = F.one_hot(indices, num_classes=self.codebook_size).type(flat_inputs.dtype)
        # EMA codebook update
        self.codebook.ema_update(encodings, flat_inputs)

        # Quantized outputs with custom gradient mapping
        e = RotationRescaleQuantizeFn.apply(flat_inputs, self.codebook().detach(), indices)
        z_q = e.reshape(B, H, W, D).permute(0, 3, 1, 2).contiguous()

        # Losses: commitment and codebook usage
        # Use stop-gradient on z_q in the commitment term to prevent decoder gradients altering encoder through this path
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())
        # Codebook loss term (optional for EMA variant): encourage codebook to move toward encoder outputs
        codebook_loss = F.mse_loss(z_q, z_e.detach())

        # Usage statistics
        usage = encodings.mean(dim=0)  # [K]

        return z_q, {
            'commitment_loss': commitment_loss,
            'codebook_loss': codebook_loss,
            'usage': usage
        }

class VQVAE_RR(nn.Module):
    """
    VQ-VAE model with Rotation-Rescale quantization and angle-preserving gradient propagation.

    Components implemented:
    - Encoder: CNN + ResidualStack
    - VectorQuantizerRR: Nearest neighbor selection, Householder-based rotation-rescale gradient mapping, EMA codebook
    - Decoder: CNN + ResidualStack

    Key parameters:
    - codebook_size: number of code vectors
    - beta: commitment loss coefficient
    - ema_decay: EMA decay for codebook updates

    Constraints:
    - Rotation/rescaling are treated as constants w.r.t gradients (stop-gradient) by detaching in autograd function.
    - Householder reflection used for computationally efficient alignment.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, embedding_dim: int = 64,
                 codebook_size: int = 1024, ema_decay: float = 0.99, beta: float = 0.25):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, hidden_channels=hidden_channels, embedding_dim=embedding_dim)
        self.quantizer = VectorQuantizerRR(codebook_size=codebook_size, embedding_dim=embedding_dim,
                                           beta=beta, ema_decay=ema_decay)
        self.decoder = Decoder(out_channels=in_channels, hidden_channels=hidden_channels, embedding_dim=embedding_dim)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, q_info = self.quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, q_info
