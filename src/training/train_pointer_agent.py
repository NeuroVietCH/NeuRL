"""Train the pointer agent on one fold (OASIS fold CSV, or any `scan_table`): cross-fitted readers, then PPO (stage A or joint stage B).

    .venv/bin/python src/training/train_pointer_agent.py --config configs/pointer_agent.yaml \
        [--fold 0] [--updates N] [--budget K] [--seed S] [--joint] [--readers <run>/readers.pt] [--name NAME]

Steps:
    1. fold train split -> train / val by subject; fold test split held out until the end
    2. train scans -> K cross-fitting groups by subject; reader i never trains on group i
    3. pretrain readers on random token subsets (or load --readers), report the random-looks baseline
    4. PPO: training scans are scored by their out-of-group reader; val/test by the ensemble
       --joint: readers keep training on the agent's picks mixed with random subsets
    5. best.pt (val `select_metric`) evaluated once on the test split

Writes to `<training.out_dir>/<name>/`: config.yaml, train.log, readers.pt, reader_history.csv,
readers_report.json, train_metrics.csv, val_metrics.csv, last.pt, best.pt, test_metrics.json, test_traces.npz.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import fields
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from agent.pointer_policy import PointerActorCritic  # noqa: E402
from env.data import load_split_samples, token_manifest  # noqa: E402
from env.pointer_env import PointerEnv  # noqa: E402
from env.token_bank import TokenBank  # noqa: E402
from evaluation.pointer_eval import BlindPolicy, evaluate_pointer  # noqa: E402
from training.common import CsvLogger, assign_groups, describe, setup_logging, split_train_val  # noqa: E402
from training.pointer_ppo import PointerPPOConfig, ppo_update, rollout  # noqa: E402
from training.reader_training import (  # noqa: E402
    FixedSets,
    ReaderConfig,
    balanced_draw,
    build_readers,
    make_reader_optimizers,
    pretrain_readers,
    reader_report,
    update_readers,
)

log = logging.getLogger("train_pointer_agent")


def dataclass_from(cls, cfg: dict):
    names = {f.name for f in fields(cls)}
    kwargs = {k: (tuple(v) if isinstance(v, list) else v) for k, v in cfg.items() if k in names}
    return cls(**kwargs)


def save_checkpoint(path: Path, policy, readers, update: int, cfg: dict, metrics: dict | None) -> None:
    torch.save({"policy": policy.state_dict(), "readers": readers.state_dict(), "update": update, "config": cfg, "metrics": metrics}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(REPO / "configs" / "pointer_agent.yaml"))
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None, help="also changes the val split and cross-fitting groups")
    parser.add_argument("--joint", action="store_true", help="stage B: keep training readers during PPO")
    parser.add_argument("--readers", default=None, help="readers.pt from an earlier run with the same fold/seed (skips pretraining)")
    parser.add_argument("--name", default=None, help="run directory name (default: <timestamp>_fold<k>_b<budget>[_joint])")
    parser.add_argument("--token_dir", default=None, help="overrides token_dir, e.g. .../neurovfm_tokens_raw_global (unstripped)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.token_dir is not None:
        cfg["token_dir"] = args.token_dir
    if args.fold is not None:
        cfg["fold"] = args.fold
    if args.updates is not None:
        cfg["ppo"]["updates"] = args.updates
    if args.budget is not None:
        cfg["environment"]["budget"] = args.budget
    if args.joint:
        cfg["joint"]["enabled"] = True
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed, train_cfg, env_cfg, joint_cfg = int(cfg["seed"]), cfg["training"], cfg["environment"], cfg["joint"]
    reader_cfg = dataclass_from(ReaderConfig, cfg["reader"])
    ppo_cfg = dataclass_from(PointerPPOConfig, cfg["ppo"])
    budget = int(env_cfg["budget"])
    if budget > reader_cfg.max_set:
        raise ValueError(f"budget {budget} > reader.max_set {reader_cfg.max_set}: readers never saw sets that large")
    torch.manual_seed(seed)
    np.random.seed(seed)

    name = args.name or f"{datetime.now():%Y%m%d_%H%M%S}_fold{cfg['fold']}_b{budget}" + ("_joint" if joint_cfg["enabled"] else "")
    run_dir = REPO / train_cfg["out_dir"] / name
    run_dir.mkdir(parents=True, exist_ok=False)
    setup_logging(log, run_dir)
    yaml.safe_dump(cfg, open(run_dir / "config.yaml", "w"), sort_keys=False)
    log.info(f"run dir: {run_dir}")
    log.info(f"data: {cfg.get('scan_table') or cfg['folds_csv']}, tokens from {cfg['token_dir']}")

    # 1. Data.
    train_m = token_manifest(cfg, fold=cfg["fold"], split="train")
    test_m = token_manifest(cfg, fold=cfg["fold"], split="test")
    val_m = token_manifest(cfg, fold=cfg["fold"], split="val")
    if len(val_m) == 0:  # no explicit val split: subject-grouped hold-out from train
        train_m, val_m = split_train_val(train_m, float(train_cfg["val_fraction"]), seed)
    t0 = time.time()
    parts = {name: load_split_samples(m) for name, m in (("train", train_m), ("val", val_m), ("test", test_m))}
    samples = parts["train"] + parts["val"] + parts["test"]
    bank = TokenBank(samples, device=cfg["device"])
    sizes = np.cumsum([0] + [len(parts[k]) for k in ("train", "val", "test")])
    dev = bank.device
    splits = {k: torch.arange(sizes[i], sizes[i + 1], device=dev) for i, k in enumerate(("train", "val", "test"))}
    for k in ("train", "val", "test"):
        log.info(describe(k, bank.labels[splits[k]].cpu().numpy()))
    log.info(f"loaded {len(bank)} scans, {len(bank.emb)} tokens in {time.time() - t0:.0f}s")

    # 2. Cross-fitting groups over training subjects.
    groups = torch.full((len(bank),), -1, dtype=torch.long, device=dev)
    train_labels = np.asarray([s["label"] for s in parts["train"]])
    train_subjects = np.asarray([s["subject_id"] for s in parts["train"]])
    groups[splits["train"]] = torch.as_tensor(assign_groups(train_labels, train_subjects, reader_cfg.k, seed), device=dev)
    group_of = {sid: int(g) for sid, g in zip(bank.sample_ids, groups.cpu().tolist(), strict=True)}

    # 3. Readers.
    readers = build_readers(reader_cfg, bank.embedding_dim).to(dev)
    if args.readers:
        ckpt = torch.load(args.readers, map_location=dev, weights_only=False)
        if ckpt["group_of"] != group_of:
            raise ValueError(f"{args.readers} was trained with different scans or cross-fitting groups (fold/seed/val split)")
        readers.load_state_dict(ckpt["readers"])
        log.info(f"readers loaded from {args.readers}")
    else:
        log.info(f"pretraining {reader_cfg.k} cross-fitted readers on random subsets of size 1..{reader_cfg.max_set}")
        history = pretrain_readers(readers, bank, splits["train"], groups, splits["val"], reader_cfg, seed, log)
        hist_csv = CsvLogger(run_dir / "reader_history.csv")
        for row in history:
            hist_csv.write(row)
    torch.save({"readers": readers.state_dict(), "group_of": group_of, "reader_config": cfg["reader"]}, run_dir / "readers.pt")
    report = reader_report(readers, bank, splits["train"], groups, {"val": splits["val"], "test": splits["test"]},
                           reader_cfg.eval_sizes, reader_cfg.eval_draws, seed)
    json.dump(report, open(run_dir / "readers_report.json", "w"), indent=2)
    log.info("random looks + readers, AUC by k: " + " | ".join(
        f"k={k}: oof {report[f'oof_auc_k{k}']:.3f} val {report[f'val_auc_k{k}']:.3f} test {report[f'test_auc_k{k}']:.3f}" for k in reader_cfg.eval_sizes))

    # 4. PPO.
    env = PointerEnv(bank, readers, budget=budget, look_cost=env_cfg["look_cost"], loss_clamp=env_cfg["loss_clamp"],
                     allow_stop=bool(env_cfg.get("allow_stop", True)))
    policy = PointerActorCritic(bank.embedding_dim, **cfg["policy"]).to(dev)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.lr)
    reader_opts = make_reader_optimizers(readers, float(joint_cfg["lr"]), reader_cfg.weight_decay) if joint_cfg["enabled"] else None
    guard = None
    if joint_cfg["enabled"] and joint_cfg.get("guard", True):
        guard = FixedSets(bank, splits["val"], reader_cfg.eval_sizes, reader_cfg.eval_draws, seed + 11)
    gen = torch.Generator(device=dev).manual_seed(seed + 7)
    n_updates, eval_every = int(cfg["ppo"]["updates"]), int(train_cfg["eval_every"])
    select = str(train_cfg["select_metric"])
    train_csv, val_csv = CsvLogger(run_dir / "train_metrics.csv"), CsvLogger(run_dir / "val_metrics.csv")
    best = {"value": -float("inf"), "update": -1}
    pending: list[dict] = []
    log.info(f"PPO: {n_updates} updates x {ppo_cfg.episodes_per_update} episodes, budget {budget}, "
             f"look cost {env_cfg['look_cost']}, STOP {'on' if env.allow_stop else 'off'}, content {'on' if policy.content else 'off'}, joint readers {'on' if joint_cfg['enabled'] else 'off'}")
    t_start = time.time()
    for update in range(1, n_updates + 1):
        scans = balanced_draw(splits["train"], bank.labels, ppo_cfg.episodes_per_update, gen)
        buf = rollout(env, policy, scans, groups[scans])
        stats = ppo_update(policy, optimizer, env, buf, ppo_cfg)
        p, y = buf["p_final"], buf["labels"]
        row = {
            "update": update, "elapsed_s": round(time.time() - t_start, 1),
            "return": float((buf["reward"] * buf["active"]).sum(1).mean()),
            "looks": float(buf["t"].float().mean()), "early_stop_rate": float((buf["t"] < budget).float().mean()),
            "train_acc": float(((p >= 0.5).long() == y).float().mean()), **stats,
        }
        if joint_cfg["enabled"]:
            pending.append({k: buf[k] for k in ("scans", "order", "t")})
            if update % int(joint_cfg["every"]) == 0:
                row["reader_loss"], row["readers_reverted"] = update_readers(
                    readers, reader_opts, bank, pending, splits["train"], groups, int(joint_cfg["steps"]), int(joint_cfg["batch_size"]),
                    float(joint_cfg["random_fraction"]), reader_cfg.max_set, gen, guard, float(joint_cfg.get("guard_tolerance", 0.0)))
                pending = []
            else:
                row["reader_loss"], row["readers_reverted"] = float("nan"), float("nan")
        train_csv.write(row)
        elapsed = time.time() - t_start
        log.info(f"[{update:>4}/{n_updates}] return {row['return']:+.3f} | looks {row['looks']:.1f} | stop early {row['early_stop_rate']:.2f} "
                 f"| train acc {row['train_acc']:.3f} | entropy {stats['entropy']:.2f} | kl {stats['approx_kl']:.4f} "
                 f"| v {stats['value_loss']:.4f} | {elapsed / 60:.1f}m, eta {elapsed / update * (n_updates - update) / 60:.1f}m")

        if update % eval_every == 0 or update == n_updates:
            metrics, _ = evaluate_pointer(env, policy, splits["val"], random_draws=int(train_cfg["random_draws"]), seed=seed)
            val_csv.write({"update": update, "elapsed_s": round(elapsed, 1), **metrics})
            is_best = metrics[select] > best["value"]
            if is_best:
                best.update(value=metrics[select], update=update)
                save_checkpoint(run_dir / "best.pt", policy, readers, update, cfg, metrics)
            save_checkpoint(run_dir / "last.pt", policy, readers, update, cfg, metrics)
            log.info(f"  VAL return {metrics['mean_return']:+.3f} vs random {metrics['random_return_matched']:+.3f} | auc {metrics['auc']:.3f} vs random {metrics['random_auc_matched']:.3f} (gain {metrics['auc_gain']:+.3f}) "
                     f"| bal acc {metrics['balanced_accuracy']:.3f} (CN {metrics['recall_cn']:.3f} / AD {metrics['recall_ad']:.3f}) "
                     f"| looks {metrics['mean_looks']:.1f}" + ("  * best" if is_best else f"  (best {select} {best['value']:.3f} @ {best['update']})"))

    # 5. Test with best.pt (policy and the readers saved with it).
    ckpt = torch.load(run_dir / "best.pt", map_location=dev, weights_only=False)
    policy.load_state_dict(ckpt["policy"])
    readers.load_state_dict(ckpt["readers"])
    test_metrics, traces = evaluate_pointer(env, policy, splits["test"], random_draws=int(train_cfg["random_draws"]), seed=seed)
    blind_metrics, _ = evaluate_pointer(env, BlindPolicy(policy), splits["test"], random_draws=int(train_cfg["random_draws"]), seed=seed)
    test_metrics = {"best_update": ckpt["update"], **test_metrics, **{f"blind_{k}": v for k, v in blind_metrics.items()}}
    json.dump(test_metrics, open(run_dir / "test_metrics.json", "w"), indent=2)
    np.savez(run_dir / "test_traces.npz", sample_id=np.asarray(bank.sample_ids)[traces["scans"]], **traces)
    log.info(f"TEST (best.pt @ update {ckpt['update']}): return {test_metrics['mean_return']:+.3f} vs random {test_metrics['random_return_matched']:+.3f} | auc {test_metrics['auc']:.3f} vs random {test_metrics['random_auc_matched']:.3f} "
             f"(gain {test_metrics['auc_gain']:+.3f}) | bal acc {test_metrics['balanced_accuracy']:.3f} "
             f"(CN {test_metrics['recall_cn']:.3f} / AD {test_metrics['recall_ad']:.3f}) | looks {test_metrics['mean_looks']:.1f}")
    log.info(f"TEST blind (same policy, seen content hidden): auc {test_metrics['blind_auc']:.3f} vs random {test_metrics['blind_random_auc_matched']:.3f} "
             f"| looks {test_metrics['blind_mean_looks']:.1f}")
    log.info(f"done in {(time.time() - t_start) / 60:.1f}m -> {run_dir}")


if __name__ == "__main__":
    main()
