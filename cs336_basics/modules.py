#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional

from torch import nn
import torch

from .functions import gelu, scaled_dot_product_attention


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
    ):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(-1) == self.d_model, \
            f"Input dimension mismatch: got {x.size(-1)}, expected {self.d_model}"

        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight


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


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: Optional[float]):
        super(MultiheadSelfAttention, self).__init__()
        assert d_model % num_heads == 0, \
            f"Division error: d_model({d_model}) should be divisible by num_heads({num_heads})"
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.q_heads = nn.ModuleList([ nn.Linear(d_model, self.d_k, bias=False) for i in range(num_heads) ])
        self.k_heads = nn.ModuleList([ nn.Linear(d_model, self.d_k, bias=False) for i in range(num_heads) ])
        self.v_heads = nn.ModuleList([ nn.Linear(d_model, self.d_v, bias=False) for i in range(num_heads) ])
        self.output_proj = nn.Linear(num_heads * self.d_v, d_model, bias=False)

    def forward(self, x: torch.Tensor):
        seq_len = x.shape[-2]
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(torch.bool)
        heads = list()
        for i in range(self.num_heads):
            q = self.q_heads[i](x)
            k = self.k_heads[i](x)
            v = self.v_heads[i](x)
            heads.append(scaled_dot_product_attention(k, q, v, mask, self.attn_pdrop))
        out = torch.cat(heads, dim=-1)
        out = self.output_proj(out)
        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 attn_pdrop: float, residual_pdrop: float):
        super(TransformerBlock, self).__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.residual_dropout = nn.Dropout(residual_pdrop)
        self.ln2 = RMSNorm(d_model)
        self.ffn = PositionWiseFFN(d_model, d_ff)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.ln1(x)
        y = self.attn(y)
        y = self.residual_dropout(y) + x
        z = self.ln2(y)
        z = self.ffn(z)
        z = self.residual_dropout(z) + y
        return z
