"""Run the standard baseline for one (dataset, encoder) pair from configs/baselines.yaml.

    .venv/bin/python scripts/run_baseline.py --dataset oasis --encoder neurovfm_local [--gpus 1] [--jobs 3]

Stages (default: all, in this order; finished pieces are skipped, so rerunning resumes):
    check    scans per fold/split/label that have a token file, embedding width, tokens per scan
    probe    logistic regression on the mean of k random tokens (src/evaluation/random_k_baseline.py)
    pointer  the pointer agent with frozen cross-fitted readers, every fold x seed x budget
             (src/training/train_pointer_agent.py); per (fold, seed) the first budget pretrains the
             readers and the others reuse them
    report   summary.csv, per_run.csv and REPORT.md, plus the cross-baseline outputs/baselines/INDEX.md

Everything goes to outputs/baselines/<dataset>__<encoder>/. The protocol (config, seeds, budgets,
probe) comes from the registry, so every dataset/encoder gets the same baseline. See docs/baselines.md.

On a SLURM cluster, run one (fold, seed) chain per array task with `--job FOLD SEED` (GPU taken
from the scheduler), then `--stages probe` and `--stages report` once.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from env.data import token_manifest  # noqa: E402

STAGES = ("check", "probe", "pointer", "report")


def repo_path(p: str | None) -> str | None:
    return None if p is None else str(p if Path(p).is_absolute() else REPO / p)


def resolve(args) -> tuple[dict, dict, Path]:
    """Registry entry -> (run config for the training/probe scripts, run info, output root)."""
    reg = yaml.safe_load(open(args.registry))
    datasets = reg["datasets"]
    if args.dataset not in datasets:
        sys.exit(f"unknown dataset {args.dataset!r}; registered: {', '.join(datasets)}")
    ds = datasets[args.dataset]
    caches = ds.get("caches") or {}
    token_dir = args.token_dir or caches.get(args.encoder)
    if token_dir is None:
        ready = [k for k, v in caches.items() if v]
        sys.exit(f"no token dir for {args.dataset}/{args.encoder}: set datasets.{args.dataset}.caches.{args.encoder} in {args.registry} "
                 f"or pass --token_dir (ready: {', '.join(ready) or 'none'})")
    token_dir = repo_path(token_dir)
    if not Path(token_dir).is_dir():
        sys.exit(f"token dir does not exist: {token_dir}")
    scan_table = repo_path(args.scan_table or ds.get("scan_table"))
    if scan_table is None and not ds.get("folds_csv"):
        sys.exit(f"{args.dataset} has neither scan_table nor folds_csv in {args.registry}")

    proto = reg["protocol"]
    cfg = yaml.safe_load(open(repo_path(proto["config"])))
    cfg["token_dir"] = token_dir
    if scan_table:
        cfg["scan_table"] = scan_table
        for k in ("folds_csv", "metadata_csv", "data_root"):
            cfg.pop(k, None)
    else:
        cfg.update({k: ds[k] for k in ("folds_csv", "metadata_csv", "data_root")})
    root = Path(repo_path(args.out_root)) / f"{args.dataset}__{args.encoder}"
    cfg["training"]["out_dir"] = str(root / "pointer")
    info = {
        "dataset": args.dataset, "encoder": args.encoder, "token_dir": token_dir,
        "scans": scan_table or ds["folds_csv"], "protocol_config": proto["config"],
        "folds": args.folds or ds.get("folds", [0]), "seeds": args.seeds or proto["seeds"], "budgets": args.budgets or proto["budgets"],
        "probe_ks": proto["probe_ks"], "probe_draws": proto["probe_draws"],
    }
    return cfg, info, root


def git_commit() -> str:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True, text=True).stdout.strip()
        return sha + ("+dirty" if dirty else "")
    except OSError:
        return "unknown"


# ---------------------------------------------------------------- check

def check(cfg: dict, info: dict, root: Path) -> dict:
    rows, first = [], None
    for fold in info["folds"]:
        for split in ("train", "val", "test"):
            m = token_manifest(cfg, fold=fold, split=split)
            rows.append({"fold": fold, "split": split, "scans": len(m), "ad": int(m["label"].sum()) if len(m) else 0,
                         "subjects": m["patient_id"].nunique() if len(m) else 0})
            if first is None and len(m):
                first = m
    table = pd.DataFrame(rows)
    problems = []
    for fold in info["folds"]:
        for split in ("train", "test"):
            r = table[(table.fold == fold) & (table.split == split)].iloc[0]
            if r.scans == 0 or r.ad == 0 or r.ad == r.scans:
                problems.append(f"fold {fold} {split}: {r.scans} scans, {r.ad} AD (needs both classes)")
    if first is None:
        sys.exit(f"no scan has a token file in {info['token_dir']}")
    sample = [torch.load(p, map_location="cpu", weights_only=False) for p in first["token_path"].iloc[:20]]
    for key in ("embeddings", "coords", "grid_shape"):
        if key not in sample[0]:
            problems.append(f"token files lack `{key}` (keys: {sorted(sample[0])})")
    counts = [len(s["embeddings"]) for s in sample]
    dim = int(sample[0]["embeddings"].shape[1])
    n_scans = len(pd.concat([token_manifest(cfg, fold=info["folds"][0], split=s) for s in ("train", "val", "test")]))
    est_gb = n_scans * float(np.median(counts)) * dim * 4 / 1e9 * 1.4  # float32 load + float16 bank
    est_gpu_gb = n_scans * float(np.median(counts)) * dim * 2 / 1e9 * 2 + 1.5  # float16 bank, its gathers, models (~4.8 GB for OASIS rawL)
    res = {"splits": rows, "embedding_dim": dim, "tokens_per_scan_sample": [int(min(counts)), int(np.median(counts)), int(max(counts))],
           "scans_per_fold": n_scans, "est_ram_gb_per_job": round(est_gb, 1), "est_gpu_gb_per_job": round(est_gpu_gb, 1), "problems": problems}
    json.dump(res, open(root / "check.json", "w"), indent=2)
    print(f"check: {info['dataset']}/{info['encoder']}, embeddings {dim}-d, tokens per scan (first 20) "
          f"min {min(counts)} / median {int(np.median(counts))} / max {max(counts)}, ~{est_gb:.1f} GB RAM and ~{est_gpu_gb:.1f} GB GPU per job")
    print(table.pivot(index="fold", columns="split", values="scans").reindex(columns=["train", "val", "test"]).to_string())
    for p in problems:
        print(f"  PROBLEM: {p}")
    return res


def gpu_free_gb() -> dict[str, float]:
    """Free memory per GPU index from nvidia-smi ({} if unavailable, e.g. under a scheduler)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
        return {i.strip(): int(m) / 1024 for i, m in (line.split(",") for line in out.strip().splitlines())}
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}


def check_gpus(gpus: list[str], jobs: int, need_gb: float) -> list[str]:
    """Problems if the chosen GPUs cannot hold their share of the concurrent jobs."""
    free = gpu_free_gb()
    problems = []
    for g in gpus:
        share = sum(1 for slot in range(jobs) if gpus[slot % len(gpus)] == g)
        if g in free and free[g] < share * need_gb:
            problems.append(f"GPU {g} has {free[g]:.1f} GB free, {share} job(s) need ~{share * need_gb:.1f} GB")
    if problems:
        roomy = [f"{i} ({f:.0f} GB)" for i, f in sorted(free.items(), key=lambda x: -x[1]) if f >= need_gb]
        problems.append(f"GPUs with room for one job: {', '.join(roomy) or 'none'}; pick with --gpus, lower --jobs, or pass --skip_gpu_check")
    return problems


# ---------------------------------------------------------------- probe / pointer

def run_logged(cmd: list[str], log_path: Path, env: dict | None = None) -> bool:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as f:
        return subprocess.run(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT, env=env).returncode == 0


def probe(cfg_path: Path, info: dict, root: Path, say) -> bool:
    out = root / "probe"
    if (out / "summary.csv").exists():
        say("skip probe (done)")
        return True
    say("start probe")
    ok = run_logged([sys.executable, "src/evaluation/random_k_baseline.py", "--config", str(cfg_path), "--out_dir", str(out),
                     "--folds", *map(str, info["folds"]), "--ks", *map(str, info["probe_ks"]), "--draws", str(info["probe_draws"])],
                    root / "logs" / "probe.log")
    say(("ok" if ok else "FAIL") + f" probe (log: {root / 'logs' / 'probe.log'})")
    return ok


def run_name(fold: int, seed: int, budget: int) -> str:
    return f"fold{fold}_s{seed}_b{budget}"


def chain(cfg_path: Path, info: dict, root: Path, fold: int, seed: int, gpu: str | None, say) -> bool:
    """All budgets of one (fold, seed); the first budget pretrains the readers, the others reuse them."""
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    readers = None
    for budget in info["budgets"]:
        name = run_name(fold, seed, budget)
        run_dir = root / "pointer" / name
        if not (run_dir / "test_metrics.json").exists():
            shutil.rmtree(run_dir, ignore_errors=True)
            say(f"start {name}" + (f" (GPU {gpu})" if gpu is not None else ""))
            cmd = [sys.executable, "src/training/train_pointer_agent.py", "--config", str(cfg_path),
                   "--fold", str(fold), "--seed", str(seed), "--budget", str(budget), "--name", name]
            if readers:
                cmd += ["--readers", readers]
            if not run_logged(cmd, root / "logs" / f"{name}.log", env):
                say(f"FAIL {name} (log: {root / 'logs' / f'{name}.log'})")
                return False
            say(f"ok {name}")
        readers = readers or str(run_dir / "readers.pt")
    return True


# ---------------------------------------------------------------- report

def md_table(df: pd.DataFrame) -> str:
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    body = [[str(i)] + [str(v) for v in row] for i, row in zip(df.index, df.to_numpy(), strict=True)]
    return "\n".join("| " + " | ".join(r) + " |" for r in [head, ["---"] * len(head), *body])


def paired_p(x: pd.Series) -> float:
    from scipy.stats import ttest_1samp
    return float(ttest_1samp(x, 0).pvalue) if len(x) >= 2 and x.std() > 0 else float("nan")


def report(info: dict, root: Path) -> None:
    rows = []
    for d in sorted((root / "pointer").glob("fold*_s*_b*")):
        if not (d / "test_metrics.json").exists():
            continue
        f, s, b = d.name.split("_")
        m = json.load(open(d / "test_metrics.json"))
        r = json.load(open(d / "readers_report.json"))
        rows.append({"fold": int(f[4:]), "seed": int(s[1:]), "budget": int(b[1:]), "auc": m["auc"], "random": m["random_auc_matched"],
                     "gain": m["auc_gain"], "blind": m["blind_auc"], "balanced_accuracy": m["balanced_accuracy"],
                     "recall_cn": m["recall_cn"], "recall_ad": m["recall_ad"], "return_gain": m["return_gain"], "best_update": m["best_update"],
                     **{k: v for k, v in r.items() if k.startswith("test_auc_k")}})
    expected = len(info["folds"]) * len(info["seeds"]) * len(info["budgets"])
    lines = [f"# Baseline: {info['dataset']} / {info['encoder']}", "",
             f"Generated {datetime.now():%Y-%m-%d %H:%M} by `scripts/run_baseline.py` (commit {git_commit()}).", "",
             f"- Tokens: `{info['token_dir']}`", f"- Scans: `{info['scans']}`",
             f"- Protocol: `{info['protocol_config']}`, folds {info['folds']}, seeds {info['seeds']}, budgets {info['budgets']}",
             f"- Pointer runs finished: **{len(rows)}/{expected}**" + ("" if len(rows) == expected else " (INCOMPLETE)"), ""]
    if (root / "check.json").exists():
        c = json.load(open(root / "check.json"))
        sp = pd.DataFrame(c["splits"])
        sp["cell"] = sp["scans"].astype(str) + " (" + sp["ad"].astype(str) + " AD)"
        lines += ["## Data", "", f"Embeddings {c['embedding_dim']}-d; tokens per scan (sample) min / median / max "
                  f"{' / '.join(map(str, c['tokens_per_scan_sample']))}.", "",
                  md_table(sp.pivot(index="fold", columns="split", values="cell").reindex(columns=["train", "val", "test"]).fillna("")), ""]
        if c["problems"]:
            lines += ["Problems: " + "; ".join(c["problems"]), ""]
    if rows:
        df = pd.DataFrame(rows).sort_values(["budget", "fold", "seed"])
        df.to_csv(root / "per_run.csv", index=False)
        g = df.groupby("budget")
        summary = pd.DataFrame({
            "runs": g.size(), "agent_auc": g.auc.mean(), "agent_auc_std": g.auc.std(), "random_auc": g.random.mean(),
            "gain": g.gain.mean(), "gain_std": g.gain.std(), "wins": g.gain.apply(lambda x: int((x > 0).sum())),
            "fold_p": g.apply(lambda x: paired_p(x.groupby("fold").gain.mean())), "blind_auc": g.blind.mean(),
            "agent_minus_blind": (df.auc - df.blind).groupby(df.budget).mean(), "balanced_accuracy": g.balanced_accuracy.mean(),
        })
        summary.to_csv(root / "summary.csv")
        lines += ["## Pointer agent vs random looks (test)", "",
                  "Mean over folds x seeds. Random = same readers, same number of random tokens, 10 draws. "
                  "Fold p = one-sample t-test of per-fold gain (mean over seeds); needs >= 2 folds. "
                  "Blind = same policy with seen content hidden; agent = blind means no per-scan adaptation.", "",
                  "| Budget | Runs | Agent AUC | Random | Gain (std) | Wins | Fold p | Blind | Bal. acc |", "|---|---|---|---|---|---|---|---|---|"]
        for b, r in summary.iterrows():
            lines.append(f"| {b} | {int(r.runs)} | **{r.agent_auc:.3f}** ({r.agent_auc_std:.3f}) | {r.random_auc:.3f} | {r.gain:+.3f} ({r.gain_std:.3f}) "
                         f"| {int(r.wins)}/{int(r.runs)} | {r.fold_p:.3f} | {r.blind_auc:.3f} | {r.balanced_accuracy:.3f} |")
        ks = sorted((c for c in df.columns if c.startswith("test_auc_k")), key=lambda c: int(c[10:]))
        first = df[df.budget == min(info["budgets"])]
        lines += ["", "Readers alone on k random tokens (test AUC, mean over the first-budget runs): "
                  + ", ".join(f"k={c[10:]} {first[c].mean():.3f}" for c in ks) + ".", ""]
    if (root / "probe" / "summary.csv").exists():
        p = pd.read_csv(root / "probe" / "summary.csv")
        lines += ["## Logistic regression on the mean of k random tokens (test)", "",
                  f"Mean over folds ({info['probe_draws']} draws per fold); `all` = every token.", "",
                  "| k | " + " | ".join(map(str, p["k"])) + " |", "|---" * (len(p) + 1) + "|",
                  "| AUC | " + " | ".join(f"{v:.3f}" for v in p["auc_mean"]) + " |",
                  "| Bal. acc | " + " | ".join(f"{v:.3f}" for v in p["bal_acc_mean"]) + " |", ""]
    (root / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    write_index(root.parent)


def write_index(base: Path) -> None:
    """One table over every baseline under outputs/baselines/, for side-by-side comparison."""
    rows = []
    for d in sorted(base.glob("*__*")):
        info = json.load(open(d / "baseline.json")) if (d / "baseline.json").exists() else {}
        probe_all = ""
        if (d / "probe" / "summary.csv").exists():
            p = pd.read_csv(d / "probe" / "summary.csv").set_index("k")
            probe_all = f"{p.loc['all', 'auc_mean']:.3f}" if "all" in p.index else ""
        if not (d / "summary.csv").exists():
            continue
        for b, r in pd.read_csv(d / "summary.csv", index_col=0).iterrows():
            rows.append(f"| {info.get('dataset', d.name.split('__')[0])} | {info.get('encoder', d.name.split('__')[1])} | {b} | {int(r.runs)} "
                        f"| {r.agent_auc:.3f} | {r.random_auc:.3f} | {r.gain:+.3f} | {int(r.wins)}/{int(r.runs)} | {r.blind_auc:.3f} | {probe_all} |")
    (base / "INDEX.md").write_text("\n".join([
        "# Standard baselines", "", "Regenerated by `scripts/run_baseline.py --stages report`. Details: `<dataset>__<encoder>/REPORT.md`.", "",
        "| Dataset | Encoder | Budget | Runs | Agent AUC | Random | Gain | Wins | Blind | Probe, all tokens |",
        "|---|---|---|---|---|---|---|---|---|---|", *rows, ""]))


# ---------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="a key of `datasets` in the registry, e.g. oasis, adni")
    parser.add_argument("--encoder", required=True, help="a key of that dataset's `caches`, e.g. neurovfm_local, resnet3d, neurojepa")
    parser.add_argument("--registry", default=str(REPO / "configs" / "baselines.yaml"))
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma-separated subset of {','.join(STAGES)}")
    parser.add_argument("--gpus", default=None, help="comma-separated GPU ids for the pointer jobs, e.g. 1 or 0,1 (default: inherit CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--jobs", type=int, default=3, help="concurrent jobs, the probe included (this server: 3, RAM-bound)")
    parser.add_argument("--folds", type=int, nargs="+", default=None, help="override the dataset's folds")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="override the protocol seeds (plan: 0 1 2)")
    parser.add_argument("--budgets", type=int, nargs="+", default=None, help="override the protocol budgets")
    parser.add_argument("--token_dir", default=None, help="override the registry's token dir")
    parser.add_argument("--scan_table", default=None, help="override the registry's scan table")
    parser.add_argument("--job", type=int, nargs=2, metavar=("FOLD", "SEED"), default=None, help="run only this (fold, seed) chain (SLURM array task)")
    parser.add_argument("--out_root", default="outputs/baselines")
    parser.add_argument("--skip_gpu_check", action="store_true", help="launch even if nvidia-smi shows too little free GPU memory")
    parser.add_argument("--dry_run", action="store_true", help="resolve and check the data, print the plan, run nothing")
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if bad := set(stages) - set(STAGES):
        sys.exit(f"unknown stages {bad}; choose from {STAGES}")
    if args.job:
        stages = ["pointer"]
    cfg, info, root = resolve(args)
    root.mkdir(parents=True, exist_ok=True)
    cfg_path = root / "run_config.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    if not (root / "baseline.json").exists() or not args.job:
        json.dump({**info, "commit": git_commit(), "updated": f"{datetime.now():%Y-%m-%d %H:%M}"}, open(root / "baseline.json", "w"), indent=2)

    lock = threading.Lock()
    log_file = root / "run.log"

    def say(msg: str) -> None:
        line = f"{datetime.now():%H:%M:%S} {msg}"
        with lock:
            print(line, flush=True)
            with open(log_file, "a") as f:
                f.write(line + "\n")

    if "check" in stages or args.dry_run or ("pointer" in stages and not (root / "check.json").exists()):
        res = check(cfg, info, root)
        if res["problems"]:
            sys.exit("data check failed; fix the problems above (or the registry) and rerun")
    gpus = args.gpus.split(",") if args.gpus else [None]
    if "pointer" in stages and args.gpus and not args.skip_gpu_check:
        need = json.load(open(root / "check.json"))["est_gpu_gb_per_job"]
        if problems := check_gpus(gpus, 1 if args.job else min(args.jobs, len(info["seeds"]) * len(info["folds"])), need):
            print("\n".join(f"  GPU: {p}" for p in problems))
            if not args.dry_run:
                sys.exit("not enough free GPU memory")
    chains = [tuple(args.job)] if args.job else [(f, s) for s in info["seeds"] for f in info["folds"]]
    if args.dry_run:
        n = len(chains) * len(info["budgets"])
        print(f"plan: {'probe + ' if 'probe' in stages else ''}{n} pointer runs ({len(chains)} fold/seed chains x budgets {info['budgets']}) "
              f"-> {root}; {args.jobs} at a time, GPUs {args.gpus or os.environ.get('CUDA_VISIBLE_DEVICES', 'inherited')}")
        return

    say(f"baseline {info['dataset']}/{info['encoder']} -> {root} (stages {','.join(stages)})")
    t0 = time.time()
    results = []
    if "probe" in stages or "pointer" in stages:
        slots = list(range(args.jobs))
        slot_lock = threading.Lock()

        def with_slot(fn, *a):
            with slot_lock:
                slot = slots.pop(0)
            try:
                return fn(*a, gpus[slot % len(gpus)])
            finally:
                with slot_lock:
                    slots.append(slot)

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = []
            if "probe" in stages:
                futures.append(pool.submit(with_slot, lambda gpu: probe(cfg_path, info, root, say)))
            if "pointer" in stages:
                for f, s in chains:
                    futures.append(pool.submit(with_slot, lambda gpu, f=f, s=s: chain(cfg_path, info, root, f, s, gpu, say)))
            results = [fu.result() for fu in futures]
    if "report" in stages:
        report(info, root)
    failed = results.count(False)
    say(f"done in {(time.time() - t0) / 60:.0f} min" + (f", {failed} job(s) FAILED: see {root / 'logs'}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
