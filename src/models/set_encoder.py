from __future__ import annotations

import math

import torch
from torch import nn


class FourierPosition(nn.Module):
    """Normalised 3D position ([-1, 1] per axis) -> sin/cos features at octave frequencies -> d."""

    def __init__(self, d: int, n_freq: int = 6) -> None:
        super().__init__()
        self.register_buffer("freqs", math.pi * 2.0 ** torch.arange(n_freq).float(), persistent=False)
        self.proj = nn.Linear(3 + 6 * n_freq, d)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        x = pos[..., None] * self.freqs  # [..., 3, F]
        feats = torch.cat([pos, x.sin().flatten(-2), x.cos().flatten(-2)], dim=-1)
        return self.proj(feats)


class SetEncoder(nn.Module):
    """Order-free encoder of a variable-size token set -> one vector (a learned summary token).

    Each token is LayerNorm(embedding) -> d plus its position encoding. An empty set returns the
    summary token alone, so "nothing seen yet" is a valid input.
    """

    def __init__(self, in_dim: int = 768, d: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1, n_freq: int = 6) -> None:
        super().__init__()
        self.in_norm = nn.LayerNorm(in_dim)
        self.proj = nn.Linear(in_dim, d)
        self.pos = FourierPosition(d, n_freq)
        self.summary = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d)

    def forward(self, emb: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """emb [B, L, in_dim], pos [B, L, 3], mask [B, L] (True = real token) -> [B, d]."""
        x = self.proj(self.in_norm(emb)) + self.pos(pos)
        x = torch.cat([self.summary.expand(len(x), -1, -1), x], dim=1)
        pad = torch.cat([mask.new_zeros(len(mask), 1), ~mask], dim=1)
        return self.out_norm(self.encoder(x, src_key_padding_mask=pad)[:, 0])
