from typing import Dict, Optional

import torch

from .stiefel import stiefel_reorth_


def _power_iteration(weight: torch.Tensor, iters: int = 1) -> torch.Tensor:
    """Estimate the spectral norm via power iteration (float32 safe)."""
    if weight.ndim != 2:
        raise ValueError("_power_iteration expects a 2-D weight matrix")
    work = weight.detach()
    if work.dtype not in (torch.float32, torch.float64):
        work = work.float()
    else:
        work = work.clone()
    device = work.device
    _, in_dim = work.shape
    v = torch.randn(in_dim, device=device, dtype=work.dtype)
    v_norm = torch.norm(v)
    if v_norm == 0:
        v = torch.ones_like(v)
        v_norm = torch.norm(v)
    v = v / v_norm
    sigma = torch.tensor(0.0, device=device, dtype=work.dtype)
    steps = max(1, int(iters))
    for _ in range(steps):
        u = work @ v
        u_norm = torch.norm(u)
        if u_norm <= 1e-9:
            sigma = torch.tensor(0.0, device=device, dtype=work.dtype)
            break
        u = u / u_norm
        v = work.t() @ u
        v_norm = torch.norm(v)
        if v_norm <= 1e-9:
            sigma = torch.tensor(0.0, device=device, dtype=work.dtype)
            break
        v = v / v_norm
        sigma = torch.norm(work @ v)
    return sigma.to(weight.dtype)


class QRProjector:
    """Apply a QR-based projection back onto the Stiefel manifold."""

    def __init__(self, mode: str = "columns") -> None:
        self.mode = mode

    def __call__(self, weight: torch.Tensor) -> Optional[Dict[str, float]]:
        stiefel_reorth_(weight, mode=self.mode)
        return None


class SpectralBudget:
    """Clamp the spectral norm of a matrix to a target Lipschitz budget."""

    def __init__(
        self,
        *,
        max_spectral: float,
        iters: int = 2,
        margin: float = 0.0,
        track: bool = False,
    ) -> None:
        if max_spectral <= 0:
            raise ValueError("max_spectral must be positive")
        if iters < 1:
            raise ValueError("iters must be >= 1")
        self.max_spectral = float(max_spectral)
        self.iters = int(iters)
        self.margin = float(max(margin, 0.0))
        self.track = bool(track)
        self.last_sigma: Optional[float] = None
        self.last_clipped: bool = False

    def __call__(self, weight: torch.Tensor) -> Optional[Dict[str, float]]:
        sigma = float(_power_iteration(weight, self.iters).item())
        threshold = self.max_spectral * (1.0 + self.margin)
        clipped = sigma > threshold
        if sigma > 0 and clipped:
            scale = self.max_spectral / (sigma + 1e-9)
            weight.mul_(scale)
            sigma = self.max_spectral
        self.last_sigma = sigma
        self.last_clipped = clipped
        if not self.track:
            return None
        return {
            'spectral_norm': sigma,
            'spectral_clipped': 1.0 if clipped else 0.0,
        }


__all__ = [
    "QRProjector",
    "SpectralBudget",
    "_power_iteration",
]
