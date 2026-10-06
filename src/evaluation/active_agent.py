from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def run_episode(env: object, policy: torch.nn.Module | None = None, greedy: bool = True) -> dict[str, object]:
    obs = env.reset()
    done = False
    while not done:
        if policy is None:
            action = 6
        else:
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                logits, _value = policy(obs_t)
                if greedy:
                    action = int(torch.argmax(logits, dim=-1).item())
                else:
                    action = int(torch.distributions.Categorical(logits=logits).sample().item())
        obs, _reward, done, _info = env.step(action)
    return {
        "label": env.label,
        "prediction": env.trace.prediction,
        "visited_indices": env.trace.visited_indices,
        "inspected_indices": env.trace.inspected_indices,
        "anomaly_scores": env.trace.anomaly_scores,
        "actions": env.trace.actions,
        "rewards": env.trace.rewards,
    }


def evaluate_policy(make_envs: Callable[[], list[object]], policy: torch.nn.Module) -> dict[str, float]:
    traces = [run_episode(env, policy=policy, greedy=True) for env in make_envs()]
    y_true = np.asarray([t["label"] for t in traces], dtype=int)
    y_pred = np.asarray([t["prediction"] for t in traces], dtype=int)
    explored = np.asarray([len(t["visited_indices"]) for t in traces], dtype=float)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "avg_explored_regions": float(explored.mean()),
    }
