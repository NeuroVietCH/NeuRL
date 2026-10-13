"""Evaluate the pointer agent against random looks scored by the same readers.

The fair comparison: for each scan, the same number of random tokens as the agent inspected,
read by the same reader ensemble. `auc_gain` > 0 means the agent's choice of tokens helps.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from env.pointer_env import PointerEnv
from training.pointer_ppo import rollout


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


@torch.no_grad()
def evaluate_pointer(env: PointerEnv, policy: torch.nn.Module, scans: torch.Tensor, batch_size: int = 256, random_draws: int = 3, seed: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Greedy episodes on `scans` (ensemble reader). Returns metrics and per-scan traces."""
    p, t, order = [], [], []
    for start in range(0, len(scans), batch_size):
        chunk = scans[start:start + batch_size]
        buf = rollout(env, policy, chunk, torch.full_like(chunk, -1), greedy=True)
        p.append(buf["p_final"])
        t.append(buf["t"])
        order.append(buf["order"])
    p_t, t_t, order_t = torch.cat(p), torch.cat(t), torch.cat(order)
    y = env.bank.labels[scans].cpu().numpy()
    p_np, t_np = p_t.cpu().numpy(), t_t.cpu().numpy()
    pred = (p_np >= 0.5).astype(int)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)

    gen = torch.Generator(device=env.bank.device).manual_seed(seed)
    ens = torch.full_like(scans, -1)

    def random_auc(sizes: torch.Tensor) -> float:
        aucs = []
        for _ in range(random_draws):
            idx, _mask = env.bank.random_subsets(scans, sizes, env.budget, gen)
            aucs.append(_auc(y, env.score(scans, idx, sizes, ens).sigmoid().cpu().numpy()))
        return float(np.mean(aucs))

    metrics = {
        "auc": _auc(y, p_np),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall_cn": float(recall[0]),
        "recall_ad": float(recall[1]),
        "mean_looks": float(t_np.mean()),
        "early_stop_rate": float((t_np < env.budget).mean()),
        "random_auc_matched": random_auc(t_t),
        "random_auc_budget": random_auc(torch.full_like(t_t, env.budget)),
    }
    metrics["auc_gain"] = metrics["auc"] - metrics["random_auc_matched"]
    traces = {"scans": scans.cpu().numpy(), "label": y, "p_ad": p_np, "looks": t_np, "order": order_t.cpu().numpy()}
    return metrics, traces
