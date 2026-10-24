from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.encoders import CachedTokenEncoder


def token_name(patient_id: str, visit_id: str, exam_date: str) -> str:
    return f"{patient_id}__{visit_id}__{str(exam_date).replace('/', '-')}.pt"


def remap_data_path(path: str | Path, data_root: str | Path) -> Path:
    """Map a brainet-style path (`/home/dev/brainet/data/oasis/...` or `data/oasis/...`) onto `data_root`.

    The shared NeuRL dataset mirrors brainet's `data/` directory, so everything after the last
    `data/` component is kept.
    """
    parts = Path(path).parts
    if "data" not in parts:
        return Path(path)
    i = len(parts) - 1 - parts[::-1].index("data")
    return Path(data_root).joinpath(*parts[i + 1 :])


def build_token_manifest(
    folds_csv: str | Path,
    token_dir: str | Path,
    metadata_csv: str | Path,
    fold: int = 0,
    split: str | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    """Rows of the OASIS fold CSV that have a cached NeuroVFM token file.

    With `data_root`, `volume_path`/`mask_path` are remapped from brainet paths onto it.
    Note: these OASIS tokens were extracted from N4 + HD-BET skull-stripped + brain-cropped
    volumes (`shared_n4_hdbet_crop`), not from the strict unstripped pipeline in `neuro_evidence`.
    """
    folds_csv = Path(folds_csv)
    df = pd.read_csv(folds_csv)
    df = df[df["fold"] == fold].copy()
    if split is not None:
        df = df[df["split"] == split].copy()
    if data_root is not None:
        for col in ("volume_path", "mask_path"):
            df[col] = df[col].map(lambda p: str(remap_data_path(p, data_root)))

    meta = pd.read_csv(metadata_csv)
    key = meta[["PTID", "VISCODE", "ID", "EXAMDATE"]].drop_duplicates()
    df["image_id"] = df["volume_path"].map(lambda p: Path(p).name.replace(".nii.gz", "").split("_")[-1])
    df = df.merge(key, left_on=["patient_id", "visit_id", "image_id"], right_on=["PTID", "VISCODE", "ID"], how="left")
    token_dir = Path(token_dir)
    df["token_path"] = [token_dir / token_name(pid, visit, date) for pid, visit, date in zip(df["patient_id"], df["visit_id"], df["EXAMDATE"], strict=False)]
    df = df[df["token_path"].map(Path.exists)].copy()
    return df


SCAN_TABLE_COLUMNS = ("sample_id", "subject_id", "label", "fold", "split")


def load_scan_table(
    scan_table: str | Path,
    token_dir: str | Path | None,
    fold: int = 0,
    split: str | None = None,
) -> pd.DataFrame:
    """Rows of a dataset-agnostic scan table that have a cached token file.

    The table has one row per (scan, fold) with columns `sample_id`, `subject_id`, `label`
    (0 = CN, 1 = AD), `fold`, `split` (`train`, `test`, optionally `val`), and either `token_path`
    (absolute) or `token_file` (relative to `token_dir`, so one table serves every encoder whose
    cache uses the same file names). Optional: `class_name`, `volume_path`, `mask_path`.
    Returned columns follow `build_token_manifest`, so `load_split_samples` works unchanged.
    """
    df = pd.read_csv(scan_table)
    missing = [c for c in SCAN_TABLE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{scan_table} lacks columns {missing}; see docs/baselines.md")
    df = df[df["fold"] == fold].copy()
    if split is not None:
        df = df[df["split"] == split].copy()
    if "token_path" not in df.columns:
        if "token_file" not in df.columns or token_dir is None:
            raise ValueError(f"{scan_table} needs `token_path`, or `token_file` plus a token_dir")
        df["token_path"] = [Path(token_dir) / f for f in df["token_file"]]
    df["token_path"] = df["token_path"].map(Path)
    df["session_id"], df["patient_id"] = df["sample_id"].astype(str), df["subject_id"].astype(str)
    if "class_name" not in df.columns:
        df["class_name"] = df["label"].map({0: "cn", 1: "ad"})
    for col in ("volume_path", "mask_path"):
        if col not in df.columns:
            df[col] = None
    return df[df["token_path"].map(Path.exists)].copy()


def token_manifest(cfg: dict[str, Any], fold: int = 0, split: str | None = None) -> pd.DataFrame:
    """Scans of one fold/split for a run config: its `scan_table` if set, else the OASIS fold CSV."""
    if cfg.get("scan_table"):
        return load_scan_table(cfg["scan_table"], cfg.get("token_dir"), fold=fold, split=split)
    return build_token_manifest(cfg["folds_csv"], cfg["token_dir"], cfg["metadata_csv"], fold=fold, split=split, data_root=cfg.get("data_root"))


def load_token_sample(row: pd.Series | dict[str, Any]) -> dict[str, Any]:
    enc = CachedTokenEncoder()
    data = enc.load(row["token_path"])
    return {
        "sample_id": row["session_id"],
        "subject_id": row["patient_id"],
        "label": int(row["label"]),
        "class_name": row["class_name"],
        "embeddings": data["embeddings"],
        "coords": data["coords"],
        "grid_shape": data["grid_shape"],
        "token_path": str(row["token_path"]),
        "volume_path": str(row["volume_path"]),
        "mask_path": str(row["mask_path"]) if "mask_path" in row and not pd.isna(row["mask_path"]) else None,
        # OASIS sample tokens come from N4 + HD-BET skull-stripped input (see build_token_manifest).
        "preprocessing": "n4_hdbet_skullstrip_crop",
    }


def load_split_samples(manifest: pd.DataFrame) -> list[dict[str, Any]]:
    """Every row of a (fold, split) manifest, e.g. the full train or test split."""
    return [load_token_sample(row) for _, row in manifest.iterrows()]


def load_balanced_samples(manifest: pd.DataFrame, n_per_class: int = 4, seed: int = 0) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows = []
    for label in sorted(manifest["label"].unique()):
        part = manifest[manifest["label"] == label]
        take = min(n_per_class, len(part))
        rows.append(part.sample(n=take, random_state=int(rng.integers(0, 1_000_000))))
    small = pd.concat(rows).sample(frac=1.0, random_state=seed)
    return [load_token_sample(row) for _, row in small.iterrows()]


def load_npz_token_sample(
    token_path: str | Path,
    label: int,
    class_name: str,
    min_brain_fraction: float | None = None,
    sample_id: str | None = None,
    subject_id: str | None = None,
) -> dict[str, Any]:
    """Load a neuro_evidence `.npz` token cache in the same format as `load_token_sample`.

    Keeps only tokens with HD-BET `brain_fraction >= min_brain_fraction` (None = keep all, incl.
    skull/scalp context tokens). Coords stay on the full patch grid, so `grid_shape` is the dense
    grid `volume_size_dhw // patch_size`.
    """
    from neuro_evidence.encoder import load_token_batch

    batch = load_token_batch(token_path)
    keep = np.ones(len(batch.coords), dtype=bool)
    if min_brain_fraction is not None:
        if batch.brain_fraction is None:
            raise ValueError(f"{token_path} has no brain_fraction; re-extract with a mask or pass min_brain_fraction=None")
        keep = batch.brain_fraction >= float(min_brain_fraction)
    return {
        "sample_id": sample_id or batch.sample_id,
        "subject_id": subject_id or batch.sample_id,
        "label": int(label),
        "class_name": class_name,
        "embeddings": batch.embeddings[keep].astype(np.float32),
        "coords": batch.coords[keep].astype(np.int64),
        "grid_shape": np.asarray(batch.dense_grid_shape, dtype=np.int64),
        "token_path": str(token_path),
        "volume_path": batch.scan_path,
        "mask_path": batch.metadata.get("mask_path"),
    }
