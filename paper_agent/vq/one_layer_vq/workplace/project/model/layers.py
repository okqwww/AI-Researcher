import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """
    Residual block used in VQ-VAE-like encoders/decoders.
    Adapted from common VQ-VAE implementations (e.g., vqvae-pytorch) with consistent style.
    """
    def __init__(self, in_channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return x + self.block(x)

class ResidualStack(nn.Module):
    """
    Stack of residual blocks.
    """
    def __init__(self, channels: int, num_blocks: int, hidden_channels: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResidualBlock(channels, hidden_channels) for _ in range(num_blocks)
        ])

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        return F.relu(x, inplace=False)

class Encoder(nn.Module):
    """
    CNN encoder mapping images to latent features.
    """
    def __init__(self, in_channels: int = 3, hidden_channels: int = 128, embedding_dim: int = 64):
        super().__init__()
        hc = hidden_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hc // 2, kernel_size=4, stride=2, padding=1),  # 32->16
            nn.ReLU(inplace=False),
            nn.Conv2d(hc // 2, hc, kernel_size=4, stride=2, padding=1),  # 16->8
            nn.ReLU(inplace=False),
            ResidualStack(hc, num_blocks=2, hidden_channels=hc // 2),
            nn.Conv2d(hc, embedding_dim, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """
    CNN decoder reconstructing images from latent features.
    """
    def __init__(self, out_channels: int = 3, hidden_channels: int = 128, embedding_dim: int = 64):
        super().__init__()
        hc = hidden_channels
        self.pre = nn.Sequential(
            nn.Conv2d(embedding_dim, hc, kernel_size=3, stride=1, padding=1),
            ResidualStack(hc, num_blocks=2, hidden_channels=hc // 2)
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hc, hc // 2, kernel_size=4, stride=2, padding=1),  # 8->16
            nn.ReLU(inplace=False),
            nn.ConvTranspose2d(hc // 2, out_channels, kernel_size=4, stride=2, padding=1)  # 16->32
        )

    def forward(self, x):
        x = self.pre(x)
        x = self.net(x)
        return x
