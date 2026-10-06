from __future__ import annotations

import numpy as np


class DistanceAnomalyDetector:
    """Z-scored distance from a reference embedding centroid."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = float(eps)
        self.center: np.ndarray | None = None
        self.mean_distance = 0.0
        self.std_distance = 1.0

    def fit(self, embeddings: np.ndarray) -> "DistanceAnomalyDetector":
        x = np.asarray(embeddings, dtype=np.float32)
        self.center = x.mean(axis=0)
        d = np.linalg.norm(x - self.center[None, :], axis=1)
        self.mean_distance = float(d.mean())
        self.std_distance = float(d.std() + self.eps)
        return self

    def score(self, embeddings: np.ndarray) -> np.ndarray:
        if self.center is None:
            raise RuntimeError("Call fit() before score().")
        x = np.asarray(embeddings, dtype=np.float32)
        d = np.linalg.norm(x - self.center[None, :], axis=1)
        return ((d - self.mean_distance) / self.std_distance).astype(np.float32)

    def score_one(self, embedding: np.ndarray) -> float:
        return float(self.score(np.asarray(embedding, dtype=np.float32)[None, :])[0])
