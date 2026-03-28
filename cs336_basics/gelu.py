#!/usr/bin/env python3
from __future__ import annotations

from math import sqrt

import torch


def gelu(x: torch.Tensor) -> torch.Tensor:
    return x * (torch.special.erf(x / sqrt(2)) + 1) / 2
