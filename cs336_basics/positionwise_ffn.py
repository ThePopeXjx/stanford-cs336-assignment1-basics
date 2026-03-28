#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn

from .gelu import gelu


class PositionWiseFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(PositionWiseFFN, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        x = self.w1(x)
        x = gelu(x)
        x = self.w2(x)
        return x
