"""Train the active agent with PPO on one OASIS fold, with live progress and checkpoints.

    .venv/bin/python src/training/train_agent.py --config configs/train_agent.yaml [--fold 0] [--updates N] [--name NAME]

Writes to `<training.out_dir>/<name>/`:
    config.yaml        resolved config
    train.log          same lines as the console
    train_metrics.csv  one row per PPO update (reward, train acc, losses, ...)
    val_metrics.csv    one row per validation eval
    last.pt, best.pt   policy checkpoints (best = highest validation balanced accuracy)
    test_metrics.json  best.pt evaluated once on the fold's test split at the end

Watch a running job with `tail -f <run>/train.log` or by reading the CSVs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from agent.policy import ActorCriticPolicy  # noqa: E402
from env.brain_patch_env import N_ACTIONS, BrainPatchEnv  # noqa: E402
from env.data import build_token_manifest, load_split_samples  # noqa: E402
from evaluation.active_agent import evaluate_policy  # noqa: E402
from training.common import CsvLogger, setup_logging, split_train_val  # noqa: E402
from training.common import describe as _describe  # noqa: E402
from training.ppo import PPOConfig, train_ppo  # noqa: E402

log = logging.getLogger("train_agent")


def describe(name: str, samples: list[dict]) -> str:
    return _describe(name, np.asarray([s["label"] for s in samples]))


def load_samples(manifest, name: str) -> list[dict]:
    t0 = time.time()
    samples = load_split_samples(manifest)
    log.info(f"{describe(name, samples)} loaded in {time.time() - t0:.0f}s")
    return samples


def save_checkpoint(path: Path, policy: torch.nn.Module, obs_dim: int, update: int, cfg: dict, metrics: dict | None = None) -> None:
    torch.save({"policy": policy.state_dict(), "obs_dim": obs_dim, "n_actions": N_ACTIONS, "update": update, "config": cfg, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "train_agent.yaml"))
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--name", default=None, help="run directory name (default: <timestamp>_fold<k>)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.fold is not None:
        cfg["fold"] = args.fold
    if args.updates is not None:
        cfg["ppo"]["updates"] = args.updates
    train_cfg = cfg["training"]
    seed = int(cfg["seed"])

    name = args.name or f"{datetime.now():%Y%m%d_%H%M%S}_fold{cfg['fold']}"
    run_dir = REPO / train_cfg["out_dir"] / name
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(log, run_dir)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    log.info(f"run dir: {run_dir}")
    log.info("data: OASIS-3 NeuroVFM tokens from SKULL-STRIPPED (N4 + HD-BET) volumes - stand-in data")

    # Data: fold train split -> train/val by subject; fold test split held out until the end.
    manifest_kwargs = dict(fold=cfg["fold"], data_root=cfg["data_root"])
    train_manifest = build_token_manifest(cfg["folds_csv"], cfg["token_dir"], cfg["metadata_csv"], split="train", **manifest_kwargs)
    test_manifest = build_token_manifest(cfg["folds_csv"], cfg["token_dir"], cfg["metadata_csv"], split="test", **manifest_kwargs)
    train_manifest, val_manifest = split_train_val(train_manifest, float(train_cfg["val_fraction"]), seed)
    train_samples = load_samples(train_manifest, "train")
    val_samples = load_samples(val_manifest, "val")
    test_samples = load_samples(test_manifest, "test")

    env_kwargs = cfg["environment"]
    rng = np.random.default_rng(seed)
    by_label = {label: [s for s in train_samples if s["label"] == label] for label in (0, 1)}

    def make_env() -> BrainPatchEnv:
        if train_cfg["balanced_sampling"]:
            pool = by_label[int(rng.integers(0, 2))]
        else:
            pool = train_samples
        sample = pool[int(rng.integers(0, len(pool)))]
        return BrainPatchEnv(sample, seed=int(rng.integers(0, 1_000_000)), **env_kwargs)

    def eval_envs(samples: list[dict]):
        # Fixed start positions so evals are comparable across updates.
        return lambda: [BrainPatchEnv(s, seed=100_000 + i, **env_kwargs) for i, s in enumerate(samples)]

    obs_dim = make_env().obs_dim
    ppo_cfg = PPOConfig(**{k: v for k, v in cfg["ppo"].items() if k != "updates"})
    n_updates = int(cfg["ppo"]["updates"])
    eval_every = int(train_cfg["eval_every"])
    train_csv, val_csv = CsvLogger(run_dir / "train_metrics.csv"), CsvLogger(run_dir / "val_metrics.csv")
    best = {"balanced_accuracy": -1.0, "update": -1}
    t_start = time.time()
    log.info(f"training: {n_updates} updates x {ppo_cfg.episodes_per_update} episodes, eval every {eval_every}")

    def on_update(stats: dict, policy: torch.nn.Module) -> None:
        update = int(stats["update"]) + 1
        elapsed = time.time() - t_start
        eta = elapsed / update * (n_updates - update)
        train_csv.write({"update": update, "elapsed_s": round(elapsed, 1), **{k: v for k, v in stats.items() if k != "update"}})
        log.info(
            f"[{update:>4}/{n_updates}] reward {stats['reward_mean']:+.3f} | train acc {stats['train_acc']:.3f} "
            f"| len {stats['episode_len_mean']:.1f} | timeout {stats['timeout_rate']:.2f} | entropy {stats['entropy']:.3f} "
            f"| pi {stats['policy_loss']:+.3f} v {stats['value_loss']:.3f} | {elapsed / 60:.1f}m, eta {eta / 60:.1f}m"
        )
        if update % eval_every == 0 or update == n_updates:
            metrics = evaluate_policy(eval_envs(val_samples), policy)
            val_csv.write({"update": update, "elapsed_s": round(elapsed, 1), **metrics})
            is_best = metrics["balanced_accuracy"] > best["balanced_accuracy"]
            if is_best:
                best.update(balanced_accuracy=metrics["balanced_accuracy"], update=update)
                save_checkpoint(run_dir / "best.pt", policy, obs_dim, update, cfg, metrics)
            save_checkpoint(run_dir / "last.pt", policy, obs_dim, update, cfg, metrics)
            log.info(
                f"  VAL bal acc {metrics['balanced_accuracy']:.3f} (CN {metrics['recall_cn']:.3f} / AD {metrics['recall_ad']:.3f}) "
                f"| acc {metrics['accuracy']:.3f} | steps {metrics['avg_steps']:.1f} | inspected {metrics['avg_inspected']:.1f} "
                f"| timeout {metrics['timeout_rate']:.2f}" + ("  * best" if is_best else f"  (best {best['balanced_accuracy']:.3f} @ {best['update']})")
            )

    train_ppo(make_env, obs_dim=obs_dim, n_actions=N_ACTIONS, updates=n_updates, config=ppo_cfg, seed=seed, on_update=on_update)

    # Final: best checkpoint on the untouched test split.
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    policy = ActorCriticPolicy(ckpt["obs_dim"], ckpt["n_actions"])
    policy.load_state_dict(ckpt["policy"])
    test_metrics = {"best_update": ckpt["update"], **evaluate_policy(eval_envs(test_samples), policy)}
    json.dump(test_metrics, open(run_dir / "test_metrics.json", "w"), indent=2)
    log.info(
        f"TEST (best.pt @ update {ckpt['update']}): bal acc {test_metrics['balanced_accuracy']:.3f} "
        f"(CN {test_metrics['recall_cn']:.3f} / AD {test_metrics['recall_ad']:.3f}) | acc {test_metrics['accuracy']:.3f}"
    )
    log.info(f"done in {(time.time() - t_start) / 60:.1f}m -> {run_dir}")


if __name__ == "__main__":
    main()
