"""Evaluate the pointer agent against random looks scored by the same readers.

The fair comparison: for each scan, the same number of random tokens as the agent inspected,
read by the same reader ensemble. `auc_gain` > 0 means the agent's choice of tokens helps.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score

from env.pointer_env import PointerEnv, true_label_loss
from training.pointer_ppo import rollout


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan")


@torch.no_grad()
def evaluate_pointer(env: PointerEnv, policy: torch.nn.Module, scans: torch.Tensor, batch_size: int = 256, random_draws: int = 3, seed: int = 0) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Greedy episodes on `scans` (ensemble reader). Returns metrics and per-scan traces."""
    p, t, order, ret = [], [], [], []
    for start in range(0, len(scans), batch_size):
        chunk = scans[start:start + batch_size]
        buf = rollout(env, policy, chunk, torch.full_like(chunk, -1), greedy=True)
        p.append(buf["p_final"])
        t.append(buf["t"])
        order.append(buf["order"])
        ret.append((buf["reward"] * buf["active"]).sum(1))
    p_t, t_t, order_t, ret_t = torch.cat(p), torch.cat(t), torch.cat(order), torch.cat(ret)
    y = env.bank.labels[scans].cpu().numpy()
    p_np, t_np = p_t.cpu().numpy(), t_t.cpu().numpy()
    pred = (p_np >= 0.5).astype(int)
    recall = recall_score(y, pred, labels=[0, 1], average=None, zero_division=0)

    gen = torch.Generator(device=env.bank.device).manual_seed(seed)
    ens = torch.full_like(scans, -1)

    labels = env.bank.labels[scans]
    loss0 = true_label_loss(env.score(scans, torch.full((len(scans), 1), -1, device=scans.device), torch.zeros_like(t_t), ens), labels, env.loss_clamp)

    def balanced_mean(x: torch.Tensor) -> float:
        x = x.cpu().numpy()
        return float(np.mean([x[y == c].mean() for c in (0, 1) if (y == c).any()]))

    def random_looks(sizes: torch.Tensor) -> tuple[float, float]:
        """AUC and class-balanced return of `sizes[b]` random tokens per scan, mean over draws."""
        aucs, rets = [], []
        for _ in range(random_draws):
            idx, _mask = env.bank.random_subsets(scans, sizes, env.budget, gen)
            logit = env.score(scans, idx, sizes, ens)
            aucs.append(_auc(y, logit.sigmoid().cpu().numpy()))
            rets.append(balanced_mean(loss0 - true_label_loss(logit, labels, env.loss_clamp) - env.look_cost * sizes))
        return float(np.mean(aucs)), float(np.mean(rets))

    rand_auc, rand_ret = random_looks(t_t)
    rand_auc_budget, _ = random_looks(torch.full_like(t_t, env.budget))

    metrics = {
        "auc": _auc(y, p_np),
        # Class-balanced mean return (reader loss drop - look costs): smoother than AUC on few AD scans.
        "mean_return": balanced_mean(ret_t),
        "random_return_matched": rand_ret,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall_cn": float(recall[0]),
        "recall_ad": float(recall[1]),
        "mean_looks": float(t_np.mean()),
        "early_stop_rate": float((t_np < env.budget).mean()),
        "random_auc_matched": rand_auc,
        "random_auc_budget": rand_auc_budget,
    }
    metrics["auc_gain"] = metrics["auc"] - metrics["random_auc_matched"]
    metrics["return_gain"] = metrics["mean_return"] - metrics["random_return_matched"]
    traces = {"scans": scans.cpu().numpy(), "label": y, "p_ad": p_np, "looks": t_np, "order": order_t.cpu().numpy()}
    return metrics, traces
