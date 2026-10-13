"""Reader (evidence classifier) training: random-subset pretraining, joint-stage updates, reports.

Cross-fitting: reader i only ever trains on training scans whose group != i. Every batch is
class-balanced, so the reader's prior is ~0.5 and both classes start an episode at loss ~log 2.
"""
from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from env.token_bank import TokenBank
from models.evidence_reader import ReaderEnsemble


@dataclass
class ReaderConfig:
    k: int = 5
    d: int = 128
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 256
    max_set: int = 32          # random subsets have size 0..max_set (0 = empty set: learns the ~0.5 prior)
    max_steps: int = 4000
    eval_every: int = 200
    patience: int = 5          # evals without val improvement before stopping
    eval_sizes: tuple[int, ...] = (1, 4, 8, 16, 32)
    eval_draws: int = 4


def balanced_draw(scans: torch.Tensor, labels: torch.Tensor, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """n scans from `scans`, each class drawn with probability 1/2."""
    dev = scans.device
    by_class = [scans[labels[scans] == c] for c in (0, 1)]
    cls = (torch.rand(n, device=dev, generator=generator) < 0.5).long()
    out = torch.empty(n, dtype=torch.long, device=dev)
    for c in (0, 1):
        rows = (cls == c).nonzero(as_tuple=True)[0]
        pick = torch.randint(len(by_class[c]), (len(rows),), device=dev, generator=generator)
        out[rows] = by_class[c][pick]
    return out


def _random_batch(bank: TokenBank, scans: torch.Tensor, max_set: int, generator: torch.Generator | None):
    sizes = torch.randint(0, max_set + 1, (len(scans),), device=scans.device, generator=generator)
    idx, mask = bank.random_subsets(scans, sizes, max_set, generator)
    emb, pos = bank.gather(scans, idx)
    return emb, pos, mask


def balanced_bce(logit: torch.Tensor, label: torch.Tensor) -> float:
    """Mean of the per-class mean BCE, so a few AD scans weigh as much as many CN scans."""
    loss = F.binary_cross_entropy_with_logits(logit, label.float(), reduction="none")
    return float(np.mean([loss[label == c].mean().item() for c in (0, 1) if (label == c).any()]))


class FixedSets:
    """Fixed random subsets of `scans` at several sizes, for comparable evals across steps."""

    def __init__(self, bank: TokenBank, scans: torch.Tensor, sizes: tuple[int, ...], draws: int, seed: int) -> None:
        gen = torch.Generator(device=bank.device).manual_seed(seed)
        self.bank, self.sets = bank, []
        for size in sizes:
            for _ in range(draws):
                s = torch.full((len(scans),), size, device=bank.device)
                idx, mask = bank.random_subsets(scans, s, size, gen)
                self.sets.append((size, scans, idx, mask))

    @torch.no_grad()
    def predict(self, fn) -> list[tuple[int, torch.Tensor, torch.Tensor]]:
        """[(size, scans, logit)] with fn(scans, emb, pos, mask) -> logit."""
        out = []
        for size, scans, idx, mask in self.sets:
            emb, pos = self.bank.gather(scans, idx)
            out.append((size, scans, fn(scans, emb, pos, mask)))
        return out


def pretrain_readers(readers: ReaderEnsemble, bank: TokenBank, train_scans: torch.Tensor, groups: torch.Tensor, val_scans: torch.Tensor,
                     cfg: ReaderConfig, seed: int, log: logging.Logger) -> list[dict]:
    """Train each reader on random subsets (incl. empty) of its allowed scans; early stop on val balanced BCE."""
    val_sets = FixedSets(bank, val_scans, cfg.eval_sizes, cfg.eval_draws, seed + 1)
    labels = bank.labels
    history = []
    for i, reader in enumerate(readers.readers):
        allowed = train_scans[groups[train_scans] != i]
        gen = torch.Generator(device=bank.device).manual_seed(seed + 100 + i)
        opt = torch.optim.AdamW(reader.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad = float("inf"), copy.deepcopy(reader.state_dict()), 0
        for step in range(1, cfg.max_steps + 1):
            reader.train()
            scans = balanced_draw(allowed, labels, cfg.batch_size, gen)
            emb, pos, mask = _random_batch(bank, scans, cfg.max_set, gen)
            loss = F.binary_cross_entropy_with_logits(reader(emb, pos, mask), labels[scans].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % cfg.eval_every == 0 or step == cfg.max_steps:
                reader.eval()
                preds = val_sets.predict(lambda s, e, p, m: reader(e, p, m))
                val = float(np.mean([balanced_bce(logit, labels[s]) for _, s, logit in preds]))
                history.append({"reader": i, "step": step, "train_loss": float(loss.item()), "val_bce": val})
                if val < best - 1e-4:
                    best, best_state, bad = val, copy.deepcopy(reader.state_dict()), 0
                else:
                    bad += 1
                if bad >= cfg.patience:
                    break
        reader.load_state_dict(best_state)
        reader.eval()
        log.info(f"  reader {i}: {len(allowed)} scans, stopped at step {step}, best val balanced BCE {best:.4f}")
    return history


def _auc(label: torch.Tensor, logit: torch.Tensor) -> float:
    y = label.cpu().numpy()
    return float(roc_auc_score(y, logit.float().cpu().numpy())) if 0 < y.sum() < len(y) else float("nan")


@torch.no_grad()
def reader_report(readers: ReaderEnsemble, bank: TokenBank, train_scans: torch.Tensor, groups: torch.Tensor, splits: dict[str, torch.Tensor],
                  sizes: tuple[int, ...], draws: int, seed: int) -> dict[str, float]:
    """AUC of readers on k random tokens: out-of-fold on train (reader i on group i) and ensemble on `splits`.

    This is the "random looks + reader" baseline the agent must beat at the same number of looks.
    """
    readers.eval()
    labels, out = bank.labels, {}
    oof = FixedSets(bank, train_scans, sizes, draws, seed + 2)
    for (size, scans, logit) in oof.predict(lambda s, e, p, m: readers(e, p, m, groups[s])):
        out.setdefault(f"oof_auc_k{size}", []).append(_auc(labels[scans], logit))
    for name, scans in splits.items():
        sets = FixedSets(bank, scans, sizes, draws, seed + 3)
        ens = torch.full_like(scans, -1)
        for (size, s, logit) in sets.predict(lambda s, e, p, m: readers(e, p, m, ens)):
            out.setdefault(f"{name}_auc_k{size}", []).append(_auc(labels[s], logit))
    return {k: float(np.mean(v)) for k, v in out.items()}


def make_reader_optimizers(readers: ReaderEnsemble, lr: float, weight_decay: float) -> list[torch.optim.Optimizer]:
    return [torch.optim.AdamW(r.parameters(), lr=lr, weight_decay=weight_decay) for r in readers.readers]


def update_readers(readers: ReaderEnsemble, optimizers: list[torch.optim.Optimizer], bank: TokenBank, episodes: list[dict],
                   train_scans: torch.Tensor, groups: torch.Tensor, steps: int, batch_size: int, random_fraction: float,
                   max_set: int, generator: torch.Generator | None = None, guard: FixedSets | None = None,
                   guard_tolerance: float = 0.0) -> tuple[float, int]:
    """Joint stage: train readers on the chooser's picks mixed with random subsets.

    `episodes`: rollouts with `scans`, `order`, `t`. A picked set is a random prefix order[:L],
    1 <= L <= t. Reader i trains only on scans with group != i (picks and random sets alike).
    With `guard` (fixed random val subsets), a reader's update is undone if it raises that reader's
    val balanced BCE by more than `guard_tolerance`, so joint training can't degrade the readers.
    Returns the mean training loss and the number of reverted reader updates.
    """
    scans = torch.cat([e["scans"] for e in episodes])
    order = torch.cat([e["order"] for e in episodes])
    t = torch.cat([e["t"] for e in episodes])
    keep = t >= 1
    scans, order, t = scans[keep], order[keep], t[keep]
    if order.shape[1] < max_set:
        order = torch.cat([order, order.new_full((len(order), max_set - order.shape[1]), -1)], dim=1)
    labels, losses, reverted = bank.labels, [], 0
    n_rand = int(round(batch_size * random_fraction))
    n_pick = batch_size - n_rand
    dev = bank.device
    for i, (reader, opt) in enumerate(zip(readers.readers, optimizers, strict=True)):
        rows_ok = (groups[scans] != i).nonzero(as_tuple=True)[0]
        allowed = train_scans[groups[train_scans] != i]
        if len(rows_ok) == 0:
            continue
        if guard is not None:
            before_state = copy.deepcopy(reader.state_dict())
            before_opt = copy.deepcopy(opt.state_dict())
            before = _guard_bce(reader, guard, labels)
        reader.train()
        for _ in range(steps):
            parts = []
            if n_pick:
                r = rows_ok[torch.randint(len(rows_ok), (n_pick,), device=dev, generator=generator)]
                L = (torch.rand(n_pick, device=dev, generator=generator) * t[r]).long() + 1
                mask = torch.arange(max_set, device=dev)[None, :] < L[:, None]
                parts.append((scans[r], torch.where(mask, order[r], -1), mask))
            if n_rand:
                s = balanced_draw(allowed, labels, n_rand, generator)
                sizes = torch.randint(0, max_set + 1, (n_rand,), device=dev, generator=generator)
                idx, mask = bank.random_subsets(s, sizes, max_set, generator)
                parts.append((s, idx, mask))
            s = torch.cat([p[0] for p in parts])
            idx = torch.cat([p[1] for p in parts])
            mask = torch.cat([p[2] for p in parts])
            emb, pos = bank.gather(s, idx)
            loss = F.binary_cross_entropy_with_logits(reader(emb, pos, mask), labels[s].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        reader.eval()
        if guard is not None and _guard_bce(reader, guard, labels) > before + guard_tolerance:
            reader.load_state_dict(before_state)
            opt.load_state_dict(before_opt)
            reverted += 1
    return (float(np.mean(losses)) if losses else float("nan")), reverted


def _guard_bce(reader: torch.nn.Module, guard: FixedSets, labels: torch.Tensor) -> float:
    reader.eval()
    preds = guard.predict(lambda s, e, p, m: reader(e, p, m))
    return float(np.mean([balanced_bce(logit, labels[s]) for _, s, logit in preds]))


def build_readers(cfg: ReaderConfig, in_dim: int) -> ReaderEnsemble:
    return ReaderEnsemble(cfg.k, in_dim=in_dim, d=cfg.d, layers=cfg.layers, heads=cfg.heads, dropout=cfg.dropout)

