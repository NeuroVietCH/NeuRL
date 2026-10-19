"""The reader: P(AD) from a set of inspected tokens, as K cross-fitted copies.

Reader i never trains on scans from cross-fitting group i, so a training scan in group i is scored
by reader i as if it were a new patient. Val/test scans (group -1) get the ensemble mean.
"""
from __future__ import annotations

import torch
from torch import nn

from models.set_encoder import SetEncoder


class EvidenceReader(nn.Module):
    def __init__(self, in_dim: int = 768, d: int = 128, layers: int = 2, heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.encoder = SetEncoder(in_dim, d, layers, heads, dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, emb: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(emb, pos, mask)).squeeze(-1)


class ReaderEnsemble(nn.Module):
    def __init__(self, k: int = 5, **reader_kwargs) -> None:
        super().__init__()
        self.readers = nn.ModuleList(EvidenceReader(**reader_kwargs) for _ in range(k))

    def forward(self, emb: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
        """Logit [B]: reader `groups[b]` for training scans, ensemble mean probability for groups[b] = -1."""
        all_logits = torch.stack([r(emb, pos, mask) for r in self.readers])  # [K, B]
        ens = torch.logit(all_logits.sigmoid().mean(0), eps=1e-6)
        own = all_logits.gather(0, groups.clamp(min=0)[None]).squeeze(0)
        return torch.where(groups >= 0, own, ens)
