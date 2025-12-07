import torch
import numpy as np
from scipy import linalg

try:
    from torchvision.models import inception_v3, Inception_V3_Weights
    _HAS_TV = True
except Exception:
    _HAS_TV = False

class FIDCalculator:
    """
    FID calculator using InceptionV3 pool3 (2048-d) features.
    Uses precomputed CIFAR-10 stats provided in dataset_candidate/cifar10-32x32.npz

    Offline-friendly: if pretrained Inception weights cannot be loaded (no internet),
    the calculator is disabled and returns NaN.
    """
    def __init__(self, stats_path: str, device: torch.device):
        data = np.load(stats_path)
        self.mu_ref = data['mu']
        self.sigma_ref = data['sigma']
        self.device = device
        self.enabled = False
        self.acts_list = []
        if _HAS_TV:
            try:
                # Torchvision enforces aux logits when using weights
                self.model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad_(False)
                self.enabled = True
            except Exception as e:
                print(f"[WARN] Could not load pretrained InceptionV3 weights for FID: {e}. FID will be NaN.")
                self.enabled = False
        else:
            print("[WARN] torchvision not available. FID will be NaN.")
            self.enabled = False

    @torch.no_grad()
    def get_acts(self, x: torch.Tensor) -> torch.Tensor:
        x = (x.clamp(-1, 1) + 1) / 2
        x = torch.nn.functional.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        feats = {}
        def hook_fn(module, inp, out):
            feats['pool'] = out.flatten(1)
        h = self.model.avgpool.register_forward_hook(hook_fn)
        try:
            _ = self.model(x)
            acts = feats['pool']
        finally:
            h.remove()
        return acts

    def update(self, x: torch.Tensor):
        if not self.enabled:
            return
        acts = self.get_acts(x.to(self.device))
        acts = acts.detach().cpu().numpy()
        self.acts_list.append(acts)

    def compute(self) -> float:
        if not self.enabled or len(self.acts_list) == 0:
            return float('nan')
        acts = np.concatenate(self.acts_list, axis=0)
        mu = np.mean(acts, axis=0)
        sigma = np.cov(acts, rowvar=False)
        diff = mu - self.mu_ref
        covmean, _ = linalg.sqrtm(sigma.dot(self.sigma_ref), disp=False)
        if not np.isfinite(covmean).all():
            eps = 1e-6
            offset = np.eye(sigma.shape[0]) * eps
            covmean = linalg.sqrtm((sigma + offset).dot(self.sigma_ref + offset))
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        fid = diff.dot(diff) + np.trace(sigma + self.sigma_ref - 2 * covmean)
        return float(fid)
