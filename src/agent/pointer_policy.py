"""Pointer actor-critic: memory of inspected tokens -> score every candidate position, plus STOP.

    h_t     = MLP([SetEncoder(inspected tokens), belief features])
    score_j = (W_q h_t) . MLP(pos_enc(position_j)) / sqrt(d)    # candidates: position only
    logits  = [score_1 .. score_N, stop(h_t)], inspected / padding tokens masked out

Candidate keys never use a token's embedding: content is revealed only by picking the token.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from models.set_encoder import FourierPosition, SetEncoder

MASKED = -1e9  # finite so masked actions give p = 0 without NaN entropies


class PointerActorCritic(nn.Module):
    def __init__(self, in_dim: int = 768, d: int = 128, layers: int = 2, heads: int = 4, n_belief: int = 3) -> None:
        super().__init__()
        self.d = d
        # dropout=0: PPO's ratio needs the same network at rollout and update time.
        self.memory = SetEncoder(in_dim, d, layers, heads, dropout=0.0)
        self.state = nn.Sequential(nn.Linear(d + n_belief, d), nn.GELU(), nn.Linear(d, d))
        self.key = nn.Sequential(FourierPosition(d), nn.GELU(), nn.Linear(d, d))
        self.query = nn.Linear(d, d)
        self.stop = nn.Linear(d, 1)
        self.value = nn.Linear(d, 1)

    def forward(self, obs: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Logits [B, N + 1] (last column = STOP) and value [B]."""
        h = self.memory(obs["seen_emb"], obs["seen_pos"], obs["seen_mask"])
        h = self.state(torch.cat([h, obs["belief"]], dim=-1))
        scores = torch.einsum("bd,bnd->bn", self.query(h), self.key(obs["cand_pos"])) / math.sqrt(self.d)
        scores = scores.masked_fill(~obs["cand_mask"], MASKED)
        return torch.cat([scores, self.stop(h)], dim=-1), self.value(h).squeeze(-1)
