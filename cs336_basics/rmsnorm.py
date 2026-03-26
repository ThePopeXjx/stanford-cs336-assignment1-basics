#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        weight: Optional[torch.Tensor] = None,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        if weight is None:
            self.weight = nn.Parameter(torch.ones(d_model, dtype=torch.float32))
        else:
            assert weight.shape == (d_model,), \
                f"Weight shape mismatch: got {weight.shape}, expected {(d_model,)}"
            self.weight = nn.Parameter(weight.to(dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.d_model, \
            f"Input dimension mismatch: got {x.size(-1)}, expected {self.d_model}"

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight
