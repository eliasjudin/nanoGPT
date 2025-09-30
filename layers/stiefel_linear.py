from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from manifold.modular import QRProjector, SpectralBudget


class StiefelLinear(nn.Module):
    """Drop-in ``nn.Linear`` with a modular Stiefel constraint on the weight."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        stiefel_mode: str = "columns",
        spectral_budget: Optional[float] = None,
        budget_power_iters: int = 2,
        budget_margin: float = 0.0,
        log_stats: bool = False,
        retraction: str = "qr",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.stiefel_mode = stiefel_mode

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        modules: List = []
        if spectral_budget is not None and spectral_budget > 0:
            modules.append(
                SpectralBudget(
                    max_spectral=float(spectral_budget),
                    iters=int(max(1, budget_power_iters)),
                    margin=float(max(0.0, budget_margin)),
                    track=log_stats,
                )
            )
        if retraction != "qr":
            raise ValueError(f"Unsupported Stiefel retraction: {retraction}")
        modules.append(QRProjector(mode=stiefel_mode))
        self._manifold_modules: List = modules
        self._manifold_metrics = {}

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        self.reproject_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    @torch.no_grad()
    def reproject_(self) -> "StiefelLinear":
        metrics = {}
        for module in self._manifold_modules:
            maybe = module(self.weight)
            if isinstance(maybe, dict):
                metrics.update(maybe)
        self._manifold_metrics = metrics
        return self

    @property
    def manifold_metrics(self) -> dict:
        return self._manifold_metrics

