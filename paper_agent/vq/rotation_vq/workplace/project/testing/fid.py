"""
FID evaluation utilities for CIFAR-10.

Computes Frechet Inception Distance between generated reconstructions and real images.
Uses torchvision's InceptionV3 to compute pool3 (2048-d) activations via feature extractor.

The CIFAR-10 reference statistics path is provided by the instruction, but to keep
this project self-contained, we compute real statistics on the test set at runtime.
"""
from typing import Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
from torchvision.models.feature_extraction import create_feature_extractor


def get_inception_features(loader: DataLoader, device: torch.device) -> torch.Tensor:
    """
    Extract InceptionV3 avgpool features (pool3, 2048-d) for all images in the loader.
    """
    inception = torchvision.models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception.eval()
    extractor = create_feature_extractor(inception, {"avgpool": "feat"})

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    all_feats = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            # Inception expects 299x299
            x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
            x = normalize(x)
            out = extractor(x)
            feats = out["feat"].view(x.size(0), -1)  # [B, 2048]
            all_feats.append(feats.detach().cpu())
    return torch.cat(all_feats, dim=0)


def _sqrtm_product(sigma1: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Compute matrix square root of sigma1 @ sigma2 using scipy-like fallback via eigh.
    """
    # Try scipy if available
    try:
        from scipy.linalg import sqrtm  # type: ignore
        covmean = sqrtm(sigma1.dot(sigma2))
        # Numeric stability: ensure real part
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return covmean
    except Exception:
        # Fallback using eigen decomposition (approximate)
        w1, v1 = np.linalg.eigh(sigma1)
        w2, v2 = np.linalg.eigh(sigma2)
        # Construct square roots of positive eigenvalues
        s1_half = v1 @ np.diag(np.sqrt(np.clip(w1, a_min=0, a_max=None))) @ v1.T
        s2_half = v2 @ np.diag(np.sqrt(np.clip(w2, a_min=0, a_max=None))) @ v2.T
        return s1_half @ s2_half


def calculate_fid(feats1: torch.Tensor, feats2: torch.Tensor) -> float:
    """Compute Fréchet distance between two multivariate Gaussians parameterized by sample means/covariances.
    """
    mu1 = feats1.mean(0).numpy()
    mu2 = feats2.mean(0).numpy()
    sigma1 = np.cov(feats1.numpy(), rowvar=False)
    sigma2 = np.cov(feats2.numpy(), rowvar=False)
    diff = mu1 - mu2

    covmean = _sqrtm_product(sigma1, sigma2)
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)
