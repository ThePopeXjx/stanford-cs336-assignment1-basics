#!/usr/bin/env python3
from __future__ import annotations

from math import sqrt
from typing import Optional

import torch
import torch.nn.functional as F


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * (torch.special.erf(x / sqrt(2)) + 1) / 2


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_entries, _ = torch.max(x, dim=dim, keepdim=True)
    x = x - max_entries
    numerator = torch.exp(x)
    denominator = torch.sum(numerator, dim=dim, keepdim=True)
    return numerator / denominator


def scaled_dot_product_attention(
    K: torch.Tensor,
    Q: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None) -> torch.Tensor:
    d_k = Q.shape[-1]
    dot_product = Q @ K.transpose(-1, -2) / sqrt(d_k)
    if mask is not None:
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(mask, -torch.inf)
        atten_score = softmax(dot_product + mask, -1)
    else:
        atten_score = softmax(dot_product, -1)
    if pdrop:
        atten_score = F.dropout(atten_score, pdrop)
    return atten_score @ V
