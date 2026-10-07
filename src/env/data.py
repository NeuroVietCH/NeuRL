from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.encoders import CachedTokenEncoder


def _resolve(path: str | Path, base: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else Path(base) / p


def token_name(patient_id: str, visit_id: str, exam_date: str) -> str:
    return f"{patient_id}__{visit_id}__{str(exam_date).replace('/', '-')}.pt"


def build_token_manifest(folds_csv: str | Path, token_dir: str | Path, metadata_csv: str | Path, fold: int = 0, split: str | None = None) -> pd.DataFrame:
    folds_csv = Path(folds_csv)
    base = folds_csv.parents[3] if len(folds_csv.parents) > 3 else Path.cwd()
    df = pd.read_csv(folds_csv)
    df = df[df["fold"] == fold].copy()
    if split is not None:
        df = df[df["split"] == split].copy()

    meta = pd.read_csv(metadata_csv)
    key = meta[["PTID", "VISCODE", "ID", "EXAMDATE"]].drop_duplicates()
    df["image_id"] = df["volume_path"].map(lambda p: Path(p).name.replace(".nii.gz", "").split("_")[-1])
    df = df.merge(key, left_on=["patient_id", "visit_id", "image_id"], right_on=["PTID", "VISCODE", "ID"], how="left")
    token_dir = Path(token_dir)
    df["token_path"] = [token_dir / token_name(pid, visit, date) for pid, visit, date in zip(df["patient_id"], df["visit_id"], df["EXAMDATE"], strict=False)]
    df = df[df["token_path"].map(Path.exists)].copy()
    return df


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
    }


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
    min_brain_fraction: float | None = 0.5,
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


def load_adnidod_samples(
    manifest_csv: str | Path,
    token_dir: str | Path,
    min_brain_fraction: float | None = 0.5,
    positive_classes: tuple[str, ...] = ("MCI", "AD", "Dementia"),
) -> list[dict[str, Any]]:
    """All ADNI-DOD scans in `manifest_processed.csv` that have a token cache in `token_dir`.

    Label 0 = CN, 1 = any of `positive_classes` (ADNI-DOD here is CN vs MCI, while the agent's
    actions are named CN/AD).
    """
    from neuro_evidence.encoder import sample_id_from_path

    manifest_csv, token_dir = Path(manifest_csv), Path(token_dir)
    root = Path(__file__).resolve().parents[2]
    samples = []
    for _, row in pd.read_csv(manifest_csv).iterrows():
        dx = str(row["DX"])
        if dx != "CN" and dx not in positive_classes:
            continue
        token_path = token_dir / f"{sample_id_from_path(_resolve(row['processed_path'], root))}.npz"
        if not token_path.exists():
            continue
        samples.append(load_npz_token_sample(token_path, label=int(dx != "CN"), class_name=dx, min_brain_fraction=min_brain_fraction, sample_id=f"{row['PTID']}_{row['VISCODE']}", subject_id=str(row["PTID"])))
    return samples
