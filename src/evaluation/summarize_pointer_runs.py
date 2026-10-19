"""Summarise pointer-agent runs across folds: agent vs random looks on each fold's test split.

    .venv/bin/python src/evaluation/summarize_pointer_runs.py [--glob 'sweep_fold*'] [--runs_dir outputs/pointer_runs]

Groups runs by (stage, budget) over folds and seeds, from each run's config.yaml (stage B = joint readers) and reports,
per group, the mean and std over folds of test AUC, random-looks AUC at matched looks, their gain,
the same for return, and in how many folds the agent beat random. Writes `<runs_dir>/summary_<glob>.csv`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs_dir", default=str(REPO / "outputs" / "pointer_runs"))
    parser.add_argument("--glob", default="sweep_fold*")
    args = parser.parse_args()

    rows = []
    for run in sorted(Path(args.runs_dir).glob(args.glob)):
        metrics_path = run / "test_metrics.json"
        if not metrics_path.exists():
            print(f"skip (no test_metrics.json): {run.name}")
            continue
        cfg = yaml.safe_load(open(run / "config.yaml"))
        m = json.load(open(metrics_path))
        rows.append({
            "run": run.name, "fold": cfg["fold"], "seed": cfg["seed"], "stage": "B" if cfg["joint"]["enabled"] else "A",
            "budget": cfg["environment"]["budget"], **{k: m[k] for k in (
                "auc", "random_auc_matched", "auc_gain", "mean_return", "random_return_matched", "return_gain",
                "balanced_accuracy", "mean_looks", "best_update")},
            "blind_auc": m.get("blind_auc", float("nan")),
        })
    if not rows:
        raise SystemExit("no finished runs matched")
    per_run = pd.DataFrame(rows).sort_values(["stage", "budget", "seed", "fold"])
    per_run["adapt_gain"] = per_run["auc"] - per_run["blind_auc"]  # agent vs the same policy blind to what it saw
    print(per_run.round(3).to_string(index=False))

    cols = ["auc", "random_auc_matched", "auc_gain", "blind_auc", "adapt_gain", "mean_return", "random_return_matched", "return_gain", "balanced_accuracy", "mean_looks"]
    g = per_run.groupby(["stage", "budget"])
    summary = g[cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary["folds"] = g.size()
    summary["auc_wins"] = g["auc_gain"].apply(lambda x: int((x > 0).sum()))
    summary["return_wins"] = g["return_gain"].apply(lambda x: int((x > 0).sum()))
    print("\nmean (std) over folds:")
    for (stage, budget), r in summary.iterrows():
        print(f"stage {stage} budget {budget} ({int(r['folds'])} folds): "
              f"AUC {r['auc_mean']:.3f} ({r['auc_std']:.3f}) vs random {r['random_auc_matched_mean']:.3f} -> gain {r['auc_gain_mean']:+.3f} ({r['auc_gain_std']:.3f}), wins {int(r['auc_wins'])}/{int(r['folds'])} | "
              f"return {r['mean_return_mean']:+.3f} vs random {r['random_return_matched_mean']:+.3f} -> gain {r['return_gain_mean']:+.3f} ({r['return_gain_std']:.3f}), wins {int(r['return_wins'])}/{int(r['folds'])} | "
              f"blind {r['blind_auc_mean']:.3f} (adapt gain {r['adapt_gain_mean']:+.3f}) | bal acc {r['balanced_accuracy_mean']:.3f} | looks {r['mean_looks_mean']:.1f}")
    out = Path(args.runs_dir) / f"summary_{args.glob.replace('*', '')}.csv"
    summary.to_csv(out)
    per_run.to_csv(out.with_name(out.stem + "_per_run.csv"), index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
