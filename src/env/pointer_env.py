"""Batched "pick a token to look at" episodes over a TokenBank.

Each step the agent picks one not-yet-inspected token of its scan (which inspects it) or STOP.
Before inspecting, the agent sees only candidate positions; a token's embedding is revealed when
it is picked. The reader scores the evidence after every look:

    reward_t = loss(E_{t-1}) - loss(E_t) - look_cost,   loss = -log P(true label | E), STOP = 0

so with gamma = 1 an episode's return is loss(nothing seen) - loss(final evidence) - cost * looks.
The episode ends at STOP or after `budget` looks; the diagnosis is the reader's final P(AD).
"""
from __future__ import annotations

import torch

from env.token_bank import TokenBank


def build_obs(bank: TokenBank, scans: torch.Tensor, order: torch.Tensor, t: torch.Tensor, belief: torch.Tensor, budget: int) -> dict[str, torch.Tensor]:
    """Policy input after t[b] looks: inspected tokens (order[b, :t[b]]), candidate positions, belief features.

    `order` [B, >=max(t)] holds picked token indices in pick order (-1 = none yet); `belief` [B] is the
    reader's current logit.
    """
    width = max(int(t.max()), 1)
    seen_idx = order[:, :width]
    seen_mask = torch.arange(width, device=t.device)[None, :] < t[:, None]
    seen_emb, seen_pos = bank.gather(scans, seen_idx)
    cand_pos, cand_valid = bank.candidates(scans)
    inspected = torch.zeros_like(cand_valid)
    rows = torch.arange(len(scans), device=t.device)[:, None].expand_as(seen_idx)
    inspected[rows[seen_mask], seen_idx[seen_mask]] = True
    p = belief.sigmoid()
    feats = torch.stack([p, (2 * p - 1).abs(), t.float() / budget], dim=-1)
    return {
        "seen_emb": seen_emb, "seen_pos": seen_pos, "seen_mask": seen_mask,
        "cand_pos": cand_pos, "cand_mask": cand_valid & ~inspected, "belief": feats,
    }


def true_label_loss(logit: torch.Tensor, label: torch.Tensor, clamp: float) -> torch.Tensor:
    """-log P(true label), with P clamped to [clamp, 1 - clamp] so one confident token can't swing it hugely."""
    p_true = torch.where(label == 1, logit.sigmoid(), 1 - logit.sigmoid())
    return -p_true.clamp(clamp, 1 - clamp).log()


class PointerEnv:
    def __init__(self, bank: TokenBank, readers: torch.nn.Module, budget: int = 8, look_cost: float = 0.01, loss_clamp: float = 0.01) -> None:
        self.bank = bank
        self.readers = readers
        self.budget = int(budget)
        self.look_cost = float(look_cost)
        self.loss_clamp = float(loss_clamp)

    @torch.no_grad()
    def score(self, scans: torch.Tensor, order: torch.Tensor, t: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Reader logit for the first t[b] picked tokens of each scan."""
        width = max(int(t.max()), 1)
        idx = order[:, :width]
        mask = torch.arange(width, device=t.device)[None, :] < t[:, None]
        emb, pos = self.bank.gather(scans, idx)
        was_training = self.readers.training
        self.readers.eval()
        logit = self.readers(emb, pos, mask, groups)
        self.readers.train(was_training)
        return logit

    def reset(self, scans: torch.Tensor, groups: torch.Tensor) -> None:
        """`groups[b]`: cross-fitting group of a training scan (scored by that reader), -1 = ensemble."""
        dev = self.bank.device
        self.scans, self.groups = scans.to(dev), groups.to(dev)
        self.labels = self.bank.labels[self.scans]
        n = len(self.scans)
        self.order = torch.full((n, self.budget), -1, dtype=torch.long, device=dev)
        self.t = torch.zeros(n, dtype=torch.long, device=dev)
        self.done = torch.zeros(n, dtype=torch.bool, device=dev)
        self.belief = self.score(self.scans, self.order, self.t, self.groups)
        self.loss = true_label_loss(self.belief, self.labels, self.loss_clamp)

    def observation(self) -> dict[str, torch.Tensor]:
        return build_obs(self.bank, self.scans, self.order, self.t, self.belief, self.budget)

    @torch.no_grad()
    def step(self, stop: torch.Tensor, token: torch.Tensor) -> torch.Tensor:
        """Apply STOP or a pick to every unfinished episode; returns the reward [B] (0 for finished ones)."""
        look = ~self.done & ~stop
        rows = look.nonzero(as_tuple=True)[0]
        self.order[rows, self.t[rows]] = token[rows]
        self.t = self.t + look.long()
        reward = torch.zeros(len(self.scans), device=self.bank.device)
        if len(rows):
            belief = self.score(self.scans[rows], self.order[rows], self.t[rows], self.groups[rows])
            loss = true_label_loss(belief, self.labels[rows], self.loss_clamp)
            reward[rows] = self.loss[rows] - loss - self.look_cost
            self.belief[rows], self.loss[rows] = belief, loss
        self.done = self.done | stop | (self.t >= self.budget)
        return reward

    def prediction(self) -> torch.Tensor:
        """Reader's current P(AD) [B]."""
        return self.belief.sigmoid()
