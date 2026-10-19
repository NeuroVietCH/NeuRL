"""All scans' tokens in flat GPU tensors, for batched episodes and reader training.

Positions are normalised per scan to its own token extent ([-1, 1] per axis, centred on the
brain), so they mean roughly the same place across unregistered subjects and carry no brain-size
information (atrophy shrinks the brain, which would let a model diagnose without looking).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch


class TokenBank:
    def __init__(self, samples: list[dict[str, Any]], device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        counts = np.asarray([len(s["embeddings"]) for s in samples], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = []
        for s in samples:
            c = np.asarray(s["coords"], dtype=np.float32)
            lo, hi = c.min(axis=0), c.max(axis=0)
            pos.append((c - (lo + hi) / 2) / np.maximum((hi - lo) / 2, 1.0))
        self.emb = torch.from_numpy(np.concatenate([np.asarray(s["embeddings"], dtype=np.float16) for s in samples])).to(self.device)
        self.pos = torch.from_numpy(np.concatenate(pos)).to(self.device)
        self.counts = torch.from_numpy(counts).to(self.device)
        self.offsets = torch.from_numpy(offsets).to(self.device)
        self.labels = torch.as_tensor([int(s["label"]) for s in samples], device=self.device)
        self.sample_ids = [str(s["sample_id"]) for s in samples]
        self.embedding_dim = int(self.emb.shape[1])

    def __len__(self) -> int:
        return len(self.sample_ids)

    def gather(self, scans: torch.Tensor, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Embeddings [B, L, D] (float32) and positions [B, L, 3] of tokens `idx` [B, L] of `scans` [B].

        Negative indices (padding) are clamped to token 0; callers mask them out.
        """
        flat = self.offsets[scans, None] + idx.clamp(min=0)
        return self.emb[flat].float(), self.pos[flat]

    def candidates(self, scans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Positions [B, N, 3] of every token of `scans`, padded to the batch max N, and valid mask [B, N]."""
        counts = self.counts[scans]
        ar = torch.arange(int(counts.max()), device=self.device)
        valid = ar[None, :] < counts[:, None]
        idx = torch.where(valid, ar[None, :], 0)
        return self.pos[self.offsets[scans, None] + idx], valid

    def random_subsets(self, scans: torch.Tensor, sizes: torch.Tensor, max_size: int, generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Distinct random token indices [B, max_size] per scan; mask [B, max_size] keeps the first `sizes[b]`."""
        counts = self.counts[scans]
        n = int(counts.max())
        keys = torch.rand(len(scans), n, device=self.device, generator=generator)
        keys = keys.masked_fill(torch.arange(n, device=self.device)[None, :] >= counts[:, None], 2.0)
        idx = keys.argsort(dim=1)[:, :max_size]
        if idx.shape[1] < max_size:
            idx = torch.cat([idx, idx.new_zeros(len(scans), max_size - idx.shape[1])], dim=1)
        mask = torch.arange(max_size, device=self.device)[None, :] < torch.minimum(sizes, counts)[:, None]
        return torch.where(mask, idx, -1), mask
