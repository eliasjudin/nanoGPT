import torch
import torch.nn as nn
from manifold.stiefel import stiefel_reorth_


class StiefelLinear(nn.Module):
    """
    nn.Linear drop-in whose weight is retracted to the Stiefel manifold
    (orthonormal columns by default). Works with standard optimizers;
    call .reproject_() after each optimizer step.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 stiefel_mode: str = "columns"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.stiefel_mode = stiefel_mode
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x @ self.weight.t()
        if self.bias is not None:
            y = y + self.bias
        return y

    @torch.no_grad()
    def reproject_(self):
        stiefel_reorth_(self.weight, mode=self.stiefel_mode)
        return self

