from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch


class PatchEncoder(Protocol):
    def encode(self, patch: np.ndarray) -> np.ndarray:
        ...


class PlaceholderEncoder:
    """Deterministic placeholder for future raw patch experiments."""

    def __init__(self, embedding_dim: int = 768, seed: int = 0) -> None:
        self.embedding_dim = int(embedding_dim)
        self.rng = np.random.default_rng(seed)
        self.proj: np.ndarray | None = None

    def encode(self, patch: np.ndarray) -> np.ndarray:
        x = np.asarray(patch, dtype=np.float32).reshape(-1)
        if self.proj is None or self.proj.shape[0] != x.shape[0]:
            self.proj = self.rng.normal(0.0, 1.0 / max(x.shape[0], 1), size=(x.shape[0], self.embedding_dim)).astype(np.float32)
        return x @ self.proj


class CachedTokenEncoder:
    """Adapter for already-extracted NeuroVFM token bundles."""

    def load(self, path: str | Path) -> dict[str, np.ndarray]:
        obj = torch.load(path, map_location="cpu")
        return {
            "embeddings": obj["embeddings"].float().numpy(),
            "coords": obj["coords"].long().numpy(),
            "grid_shape": obj["grid_shape"].long().numpy(),
        }
