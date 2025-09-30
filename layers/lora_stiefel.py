import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifold.stiefel import stiefel_reorth_


class LoRALinear(nn.Linear):
    """
    LoRA nn.Linear with optional Stiefel constraint on B (up-projection).

    Shapes and update:
    - A ∈ R^{r × in}, B ∈ R^{out × r}
    - Effective weight: W_eff = W + (α/r) (B @ A)

    Stiefel-B (orthonormal columns when out ≥ r) yields ||B||₂ = 1 and κ₂(B) = 1
    along the update subspace, stabilizing the low-rank update. We keep ΔW = 0 at
    initialization by zero-initializing A when stiefel_B=True.
    """
    def __init__(self, in_features, out_features, bias=True,
                 lora_rank: int = 0, lora_alpha: float = 0.0, lora_dropout: float = 0.0,
                 stiefel_B: bool = False, stiefel_mode: str = "columns",
                 device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = float(lora_alpha)
        self.lora_dropout = nn.Dropout(lora_dropout) if self.lora_rank > 0 else None
        self.scaling = (self.lora_alpha / self.lora_rank) if self.lora_rank > 0 else 0.0
        self.stiefel_B = bool(stiefel_B)
        self.stiefel_mode = stiefel_mode
        if self.lora_rank > 0:
            # Ensure A/B live on the same device/dtype as the base weight
            pdev, pdtype = self.weight.device, self.weight.dtype
            self.A = nn.Parameter(torch.empty(self.lora_rank, in_features, device=pdev, dtype=pdtype))
            self.B = nn.Parameter(torch.empty(out_features, self.lora_rank, device=pdev, dtype=pdtype))
            self.A.requires_grad = True
            self.B.requires_grad = True
            self.reset_lora()

    def reset_lora(self):
        if self.lora_rank > 0:
            if self.stiefel_B:
                # Keep ΔW = 0 at init by zeroing A and setting B ≈ Stiefel
                nn.init.zeros_(self.A)
                nn.init.orthogonal_(self.B)
                # Enforce the intended Stiefel mode for B by shape
                out, r = self.B.shape
                mode = "columns" if out >= r else "rows"
                stiefel_reorth_(self.B, mode=mode)
            else:
                # Standard LoRA: small A, zero B to keep ΔW = 0
                torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
                torch.nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.lora_rank > 0:
            u = self.lora_dropout(x) if self.lora_dropout else x
            y = y + self.scaling * F.linear(F.linear(u, self.A), self.B)
        return y

    @torch.no_grad()
    def reproject_(self):
        if self.lora_rank > 0 and self.stiefel_B:
            # Choose a feasible mode by shape to avoid projecting zero-start B to arbitrary Q
            out, r = self.B.shape
            mode = self.stiefel_mode
            if mode not in ("columns", "rows"):
                mode = "columns" if out >= r else "rows"
            # Guard: keep ΔW = 0 for zero B before first actual update
            if torch.count_nonzero(self.B).item() == 0:
                return self
            stiefel_reorth_(self.B, mode=mode)
        return self
