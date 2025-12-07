"""
VQ-VAE architecture integrating the novel Rotation-Rescaling Transform in the quantizer.

Components:
- Encoder/Decoder CNN suitable for CIFAR-10 32x32 images
- VectorQuantizerEMA with RRT straight-through gradients

This implementation is self-contained and does not import from external repositories.
"""
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from .vector_quantizer import VectorQuantizerEMA

class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, hidden: int = 128, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1),  # 32->16
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 4, 2, 1),  # 16->8
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 4, 2, 1),  # 8->4
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, latent_dim, 3, 1, 1),  # keep 4x4 latent grid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, hidden: int = 128, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, hidden, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),  # 4->8
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, hidden, 4, 2, 1),  # 8->16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden, out_channels, 4, 2, 1),  # 16->32
            nn.Sigmoid(),  # normalize to [0,1]
        )

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        return self.net(z_q)

class VQVAE(nn.Module):
    def __init__(self, img_channels: int = 3, codebook_size: int = 1024, beta: float = 0.25,
                 embedding_dim: int = 64, ema_decay: float = 0.99, use_rrt: bool = True):
        super().__init__()
        self.encoder = Encoder(in_channels=img_channels, latent_dim=embedding_dim)
        self.quantizer = VectorQuantizerEMA(embedding_dim=embedding_dim,
                                            num_embeddings=codebook_size,
                                            decay=ema_decay,
                                            beta=beta,
                                            use_rrt=use_rrt)
        self.decoder = Decoder(out_channels=img_channels, latent_dim=embedding_dim)

    def forward(self, x: torch.Tensor):
        z_e = self.encoder(x)
        z_q, indices, vq_loss, stats = self.quantizer(z_e)
        x_rec = self.decoder(z_q)
        return x_rec, vq_loss, stats
