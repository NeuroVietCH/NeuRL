from __future__ import annotations

from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import yaml


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/baseline.yaml") -> dict[str, Any]:
    cfg_path = resolve_path(path, base=repo_root())
    with cfg_path.open("r") as f:
        return yaml.safe_load(f)


def resolve_path(path: str | Path, base: str | Path | None = None) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (Path(base) if base is not None else repo_root()) / p


def config_path(config: dict[str, Any], key: str) -> Path:
    return resolve_path(config[key], base=repo_root())


def nifti_summary(path: str | Path) -> dict[str, Any]:
    img = nib.load(str(path))
    data = np.asanyarray(img.dataobj)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return {
        "path": str(path),
        "shape": tuple(int(v) for v in data.shape),
        "dtype": str(data.dtype),
        "zooms": zooms,
        "affine": img.affine,
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean": float(np.nanmean(data)),
    }


def load_nifti_array(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    img = nib.load(str(path))
    return np.asanyarray(img.dataobj), img.affine
