import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from manifold.stiefel import stiefel_reorth_


class LoRALinear(nn.Linear):
    """
    LoRA with optional Stiefel constraint on B (up-projection). Shapes match
    the common convention: A ∈ R^{r x in}, B ∈ R^{out x r}, W_eff = W + α/r * (B @ A).
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
            self.A = nn.Parameter(torch.empty(self.lora_rank, in_features))
            self.B = nn.Parameter(torch.empty(out_features, self.lora_rank))
            self.A.requires_grad = True
            self.B.requires_grad = True
            self.reset_lora()

    def reset_lora(self):
        if self.lora_rank > 0:
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
            stiefel_reorth_(self.B, mode=self.stiefel_mode)
        return self

