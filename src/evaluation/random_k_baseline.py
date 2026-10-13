"""Random-look baseline: CN/AD from the mean of k random tokens per scan, per OASIS fold.

    .venv/bin/python src/evaluation/random_k_baseline.py [--folds 0 1 2 3 4] [--draws 10] [--ks 1 2 4 8 16 32 64]

For each fold and budget k, every scan (train and test) keeps k tokens drawn uniformly without
replacement and is mean-pooled. A fresh StandardScaler + logistic-regression probe is fit on the
fold's train split and scored on its test split. Repeated over `--draws` seeds; k="all" pools every
token and runs once. This is the "k random tokens" baseline the agent must beat at the same budget.

Writes `outputs/random_k_baseline/per_draw.csv` (one row per fold/k/draw) and `summary.csv`.
With `--save_probes`, also every fitted probe to `probes/fold<f>/k<k>_draw<d>.joblib` (k="all" -> `kall_draw0`):
a dict with the sklearn pipeline (scaler + logistic regression on the mean of k tokens) and its fold/k/draw/seed/test scores.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from env.data import build_token_manifest  # noqa: E402


def pool(embeddings: list[np.ndarray], k: int | None, rng: np.random.Generator) -> np.ndarray:
    if k is None:
        return np.asarray([e.mean(0) for e in embeddings])
    return np.asarray([e[rng.choice(len(e), min(k, len(e)), replace=False)].mean(0) for e in embeddings])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "train_agent.yaml"))
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--draws", type=int, default=10)
    parser.add_argument("--C", type=float, default=0.05)
    parser.add_argument("--out_dir", default=str(REPO / "outputs" / "random_k_baseline"))
    parser.add_argument("--save_probes", action="store_true", help="save every fitted probe under <out_dir>/probes/")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}  # token_path -> embeddings; scans repeat across folds

    def load(fold: int, split: str):
        m = build_token_manifest(cfg["folds_csv"], cfg["token_dir"], cfg["metadata_csv"], fold=fold, split=split, data_root=cfg["data_root"])
        for p in map(str, m["token_path"]):
            if p not in cache:
                cache[p] = torch.load(p, map_location="cpu")["embeddings"].float().numpy()
        return [cache[str(p)] for p in m["token_path"]], m["label"].to_numpy()

    rows = []
    for fold in args.folds:
        Etr, ytr = load(fold, "train")
        Ete, yte = load(fold, "test")
        print(f"fold {fold}: train {len(ytr)} ({ytr.sum()} AD), test {len(yte)} ({yte.sum()} AD)", flush=True)
        for k in [*args.ks, None]:
            for draw in range(1 if k is None else args.draws):
                seed = 1000 * fold + draw
                rng = np.random.default_rng(seed)
                clf = make_pipeline(StandardScaler(), LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced"))
                clf.fit(pool(Etr, k, rng), ytr)
                p = clf.predict_proba(pool(Ete, k, rng))[:, 1]
                rows.append({
                    "fold": fold, "k": "all" if k is None else k, "draw": draw,
                    "n_test": len(yte), "n_test_ad": int(yte.sum()),
                    "auc": roc_auc_score(yte, p), "balanced_accuracy": balanced_accuracy_score(yte, p >= 0.5),
                })
                if args.save_probes:
                    probe_dir = out_dir / "probes" / f"fold{fold}"
                    probe_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump({"pipeline": clf, "seed": seed, "C": args.C, "config": args.config, **rows[-1]},
                                probe_dir / f"k{rows[-1]['k']}_draw{draw}.joblib")
            last = [r["auc"] for r in rows if r["fold"] == fold and r["k"] == rows[-1]["k"]]
            print(f"  k={rows[-1]['k']!s:>4}  AUC {np.mean(last):.3f} ± {np.std(last):.3f}", flush=True)

    per_draw = pd.DataFrame(rows)
    per_draw.to_csv(out_dir / "per_draw.csv", index=False)
    # Per fold: mean over draws. Across folds: mean and std of the per-fold means.
    per_fold = per_draw.groupby(["k", "fold"], sort=False)[["auc", "balanced_accuracy"]].mean().reset_index()
    draw_std = per_draw.groupby(["k", "fold"], sort=False)["auc"].std().groupby(level=0, sort=False).mean()
    summary = per_fold.groupby("k", sort=False).agg(
        auc_mean=("auc", "mean"), auc_fold_std=("auc", "std"),
        bal_acc_mean=("balanced_accuracy", "mean"), bal_acc_fold_std=("balanced_accuracy", "std"),
    )
    summary["auc_draw_std"] = draw_std
    summary.to_csv(out_dir / "summary.csv")
    print("\nper-fold AUC (mean over draws):")
    print(per_fold.pivot(index="k", columns="fold", values="auc").loc[summary.index].round(3).to_string())
    print("\nsummary over folds:")
    print(summary.round(3).to_string())


if __name__ == "__main__":
    main()
