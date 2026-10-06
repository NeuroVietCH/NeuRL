from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


MOVE_DELTAS = {
    0: np.array([1, 0, 0], dtype=np.int64),
    1: np.array([-1, 0, 0], dtype=np.int64),
    2: np.array([0, 1, 0], dtype=np.int64),
    3: np.array([0, -1, 0], dtype=np.int64),
    4: np.array([0, 0, 1], dtype=np.int64),
    5: np.array([0, 0, -1], dtype=np.int64),
}
INSPECT_ACTION = 6
DIAGNOSE_CN_ACTION = 7
DIAGNOSE_AD_ACTION = 8
N_ACTIONS = 9


@dataclass
class EvidenceItem:
    token_index: int
    coord: np.ndarray
    anomaly_score: float
    embedding: np.ndarray


@dataclass
class EpisodeTrace:
    visited_indices: list[int] = field(default_factory=list)
    inspected_indices: list[int] = field(default_factory=list)
    anomaly_scores: list[float] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    prediction: int | None = None


class BrainPatchEnv:
    """Discrete token/patch navigation environment for one MRI scan."""

    def __init__(
        self,
        sample: dict[str, Any],
        anomaly_detector: Any,
        max_steps: int = 24,
        move_penalty: float = -0.01,
        inspect_penalty: float = -0.02,
        useful_evidence_scale: float = 0.05,
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
        seed: int = 0,
    ) -> None:
        self.sample = sample
        self.embeddings = np.asarray(sample["embeddings"], dtype=np.float32)
        self.coords = np.asarray(sample["coords"], dtype=np.int64)
        self.label = int(sample["label"])
        self.grid_shape = tuple(int(v) for v in sample["grid_shape"])
        self.anomaly_detector = anomaly_detector
        self.max_steps = int(max_steps)
        self.move_penalty = float(move_penalty)
        self.inspect_penalty = float(inspect_penalty)
        self.useful_evidence_scale = float(useful_evidence_scale)
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.rng = np.random.default_rng(seed)
        self.coord_to_index = {tuple(c.tolist()): i for i, c in enumerate(self.coords)}
        self.embedding_dim = int(self.embeddings.shape[1])
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.embedding_dim + 8

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.current_index = int(self.rng.integers(0, len(self.embeddings)))
        self.evidence: list[EvidenceItem] = []
        self.done = False
        self.trace = EpisodeTrace()
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.done:
            raise RuntimeError("Episode is done. Call reset().")

        action = int(action)
        reward = 0.0
        info: dict[str, Any] = {}
        self.step_count += 1
        self.trace.actions.append(action)

        if action in MOVE_DELTAS:
            reward += self.move_penalty
            self._move(action)
        elif action == INSPECT_ACTION:
            reward += self._inspect()
        elif action in (DIAGNOSE_CN_ACTION, DIAGNOSE_AD_ACTION):
            pred = 0 if action == DIAGNOSE_CN_ACTION else 1
            self.done = True
            self.trace.prediction = pred
            reward += self.correct_reward if pred == self.label else self.incorrect_reward
            info["prediction"] = pred
        else:
            reward -= 0.1

        if self.step_count >= self.max_steps and not self.done:
            self.done = True
            pred = self.memory_prediction()
            self.trace.prediction = pred
            reward += self.correct_reward if pred == self.label else self.incorrect_reward
            info["prediction"] = pred
            info["timeout"] = True

        self.trace.visited_indices.append(self.current_index)
        self.trace.rewards.append(float(reward))
        return self._observation(), float(reward), self.done, info

    def memory_prediction(self) -> int:
        if not self.evidence:
            return 0
        return int(np.mean([e.anomaly_score for e in self.evidence]) >= 0.0)

    def _move(self, action: int) -> None:
        target = tuple((self.coords[self.current_index] + MOVE_DELTAS[action]).tolist())
        if target in self.coord_to_index:
            self.current_index = self.coord_to_index[target]

    def _inspect(self) -> float:
        emb = self.embeddings[self.current_index]
        score = float(self.anomaly_detector.score_one(emb))
        self.evidence.append(EvidenceItem(self.current_index, self.coords[self.current_index].copy(), score, emb.copy()))
        self.trace.inspected_indices.append(self.current_index)
        self.trace.anomaly_scores.append(score)
        return self.inspect_penalty + self.useful_evidence_scale * max(score, 0.0)

    def _memory_features(self) -> np.ndarray:
        if not self.evidence:
            return np.zeros(5, dtype=np.float32)
        scores = np.asarray([e.anomaly_score for e in self.evidence], dtype=np.float32)
        return np.asarray([len(self.evidence) / max(self.max_steps, 1), scores.mean(), scores.max(), scores.min(), scores[-1]], dtype=np.float32)

    def _observation(self) -> np.ndarray:
        emb = self.embeddings[self.current_index].astype(np.float32)
        coord = self.coords[self.current_index].astype(np.float32)
        denom = np.maximum(np.asarray(self.grid_shape, dtype=np.float32) - 1.0, 1.0)
        pos = coord / denom
        return np.concatenate([emb, pos, self._memory_features()], axis=0).astype(np.float32)
