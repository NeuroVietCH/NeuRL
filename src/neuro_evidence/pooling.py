from __future__ import annotations

import numpy as np


def mean_pool_tokens(embeddings: np.ndarray) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError(f"Expected [N,D] embeddings, got {embeddings.shape}")
    return embeddings.mean(axis=0).astype(np.float32)


def coordinate_summary(coords: np.ndarray, dense_grid_shape: tuple[int, int, int]) -> dict[str, object]:
    coords = coords.astype(np.int64, copy=False)
    grid = np.asarray(dense_grid_shape, dtype=np.int64)
    unique = np.unique(coords, axis=0)
    invalid = np.any(coords < 0, axis=1) | np.any(coords >= grid[None, :], axis=1)
    return {
        "coord_min": coords.min(axis=0).tolist() if coords.size else [],
        "coord_max": coords.max(axis=0).tolist() if coords.size else [],
        "unique_coordinates": int(unique.shape[0]),
        "duplicate_coordinates": int(coords.shape[0] - unique.shape[0]),
        "invalid_coordinates": int(invalid.sum()),
        "dense_grid_shape": tuple(int(v) for v in dense_grid_shape),
        "dense_token_count": int(np.prod(grid)),
        "missing_dense_coordinates": int(np.prod(grid) - unique.shape[0]),
    }
