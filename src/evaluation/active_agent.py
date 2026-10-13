from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score


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
        "actions": env.trace.actions,
        "rewards": env.trace.rewards,
    }


def evaluate_policy(make_envs: Callable[[], list[object]], policy: torch.nn.Module) -> dict[str, float]:
    traces = [run_episode(env, policy=policy, greedy=True) for env in make_envs()]
    y_true = np.asarray([t["label"] for t in traces], dtype=int)
    y_pred = np.asarray([t["prediction"] for t in traces], dtype=int)
    explored = np.asarray([len(set(t["visited_indices"])) for t in traces], dtype=float)
    recall = recall_score(y_true, y_pred, labels=[0, 1], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall_cn": float(recall[0]),
        "recall_ad": float(recall[1]),
        "avg_explored_regions": float(explored.mean()),
        "avg_steps": float(np.mean([len(t["actions"]) for t in traces])),
        "avg_inspected": float(np.mean([len(set(t["inspected_indices"])) for t in traces])),
        # Episode ended by the step limit instead of a diagnose action (7 = CN, 8 = AD).
        "timeout_rate": float(np.mean([t["actions"][-1] not in (7, 8) for t in traces])),
    }
