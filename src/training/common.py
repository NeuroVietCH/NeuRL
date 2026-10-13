"""Helpers shared by the training scripts: run logging, CSV metrics, subject-grouped splits."""
from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


class CsvLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fields: list[str] | None = None

    def write(self, row: dict) -> None:
        new = self.fields is None
        if new:
            self.fields = list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields)
            if new:
                w.writeheader()
            w.writerow(row)


def setup_logging(log: logging.Logger, run_dir: Path) -> None:
    fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    log.setLevel(logging.INFO)
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "train.log")):
        handler.setFormatter(fmt)
        log.addHandler(handler)


def split_train_val(manifest: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Subject-grouped, label-stratified hold-out of ~val_fraction of the fold's train rows."""
    n_splits = max(2, round(1.0 / val_fraction))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_idx, val_idx = next(splitter.split(manifest, manifest["label"], groups=manifest["patient_id"]))
    return manifest.iloc[train_idx], manifest.iloc[val_idx]


def assign_groups(labels: np.ndarray, subjects: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Cross-fitting groups: each scan gets a group in [0, k), subject-grouped and label-stratified."""
    groups = np.full(len(labels), -1, dtype=np.int64)
    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    for g, (_, idx) in enumerate(splitter.split(np.zeros(len(labels)), labels, groups=subjects)):
        groups[idx] = g
    return groups


def describe(name: str, labels: np.ndarray) -> str:
    labels = np.asarray(labels)
    return f"{name}: {len(labels)} scans ({int((labels == 0).sum())} CN / {int((labels == 1).sum())} AD)"
