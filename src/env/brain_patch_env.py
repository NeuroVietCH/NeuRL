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
    embedding: np.ndarray


@dataclass
class EpisodeTrace:
    visited_indices: list[int] = field(default_factory=list)
    inspected_indices: list[int] = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    prediction: int | None = None


class BrainPatchEnv:
    """Discrete token/patch navigation environment for one MRI scan.

    On timeout the episode ends with `timeout_prediction` (e.g. the training majority class).
    """

    def __init__(
        self,
        sample: dict[str, Any],
        max_steps: int = 24,
        move_penalty: float = -0.01,
        inspect_penalty: float = -0.02,
        correct_reward: float = 1.0,
        incorrect_reward: float = -1.0,
        timeout_prediction: int = 0,
        seed: int = 0,
    ) -> None:
        self.sample = sample
        self.embeddings = np.asarray(sample["embeddings"], dtype=np.float32)
        self.coords = np.asarray(sample["coords"], dtype=np.int64)
        self.label = int(sample["label"])
        self.grid_shape = tuple(int(v) for v in sample["grid_shape"])
        self.max_steps = int(max_steps)
        self.move_penalty = float(move_penalty)
        self.inspect_penalty = float(inspect_penalty)
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.timeout_prediction = int(timeout_prediction)
        self.rng = np.random.default_rng(seed)
        self.coord_to_index = {tuple(c.tolist()): i for i, c in enumerate(self.coords)}
        self.embedding_dim = int(self.embeddings.shape[1])
        self.reset()

    @property
    def obs_dim(self) -> int:
        return self.embedding_dim + 6

    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.current_index = int(self.rng.integers(0, len(self.embeddings)))
        self.evidence: list[EvidenceItem] = []
        self.inspected: set[int] = set()
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
            pred = self.timeout_prediction
            self.trace.prediction = pred
            reward += self.correct_reward if pred == self.label else self.incorrect_reward
            info["prediction"] = pred
            info["timeout"] = True

        self.trace.visited_indices.append(self.current_index)
        self.trace.rewards.append(float(reward))
        return self._observation(), float(reward), self.done, info

    def _move(self, action: int) -> None:
        target = tuple((self.coords[self.current_index] + MOVE_DELTAS[action]).tolist())
        if target in self.coord_to_index:
            self.current_index = self.coord_to_index[target]

    def _inspect(self) -> float:
        # Re-inspecting a token costs the same but adds no new evidence.
        if self.current_index not in self.inspected:
            self.inspected.add(self.current_index)
            emb = self.embeddings[self.current_index]
            self.evidence.append(EvidenceItem(self.current_index, self.coords[self.current_index].copy(), emb.copy()))
            self.trace.inspected_indices.append(self.current_index)
        return self.inspect_penalty

    def _memory_features(self) -> np.ndarray:
        horizon = max(self.max_steps, 1)
        return np.asarray(
            [len(self.evidence) / horizon, float(self.current_index in self.inspected), self.step_count / horizon],
            dtype=np.float32,
        )

    def _observation(self) -> np.ndarray:
        emb = self.embeddings[self.current_index].astype(np.float32)
        coord = self.coords[self.current_index].astype(np.float32)
        denom = np.maximum(np.asarray(self.grid_shape, dtype=np.float32) - 1.0, 1.0)
        pos = coord / denom
        return np.concatenate([emb, pos, self._memory_features()], axis=0).astype(np.float32)
