from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from agent.policy import ActorCriticPolicy


@dataclass
class PPOConfig:
    epochs: int = 4
    episodes_per_update: int = 16
    gamma: float = 0.98
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


def _discounted_returns(rewards: list[float], gamma: float) -> list[float]:
    out: list[float] = []
    running = 0.0
    for r in reversed(rewards):
        running = float(r) + gamma * running
        out.append(running)
    return list(reversed(out))


def train_ppo(make_env: Callable[[], object], obs_dim: int, n_actions: int, updates: int = 8, config: PPOConfig | None = None, seed: int = 0):
    cfg = config or PPOConfig()
    torch.manual_seed(seed)
    policy = ActorCriticPolicy(obs_dim, n_actions)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    history: list[dict[str, float]] = []

    for update in range(updates):
        obs_buf: list[np.ndarray] = []
        act_buf: list[int] = []
        logp_buf: list[float] = []
        ret_buf: list[float] = []
        val_buf: list[float] = []
        episode_rewards: list[float] = []
        episode_lengths: list[int] = []

        for _ in range(cfg.episodes_per_update):
            env = make_env()
            obs = env.reset()
            rewards: list[float] = []
            done = False
            steps = 0
            while not done:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, logp, value = policy.act(obs_t)
                next_obs, reward, done, _info = env.step(int(action.item()))
                obs_buf.append(obs)
                act_buf.append(int(action.item()))
                logp_buf.append(float(logp.item()))
                val_buf.append(float(value.item()))
                rewards.append(float(reward))
                obs = next_obs
                steps += 1
            ret_buf.extend(_discounted_returns(rewards, cfg.gamma))
            episode_rewards.append(float(sum(rewards)))
            episode_lengths.append(steps)

        obs_t = torch.as_tensor(np.asarray(obs_buf), dtype=torch.float32)
        act_t = torch.as_tensor(act_buf, dtype=torch.long)
        old_logp_t = torch.as_tensor(logp_buf, dtype=torch.float32)
        ret_t = torch.as_tensor(ret_buf, dtype=torch.float32)
        val_t = torch.as_tensor(val_buf, dtype=torch.float32)
        adv_t = ret_t - val_t
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        for _ in range(cfg.epochs):
            logits, values = policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(act_t)
            ratio = torch.exp(logp - old_logp_t)
            clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_t
            policy_loss = -torch.min(ratio * adv_t, clipped).mean()
            value_loss = nn.functional.mse_loss(values, ret_t)
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()

        history.append({"update": float(update), "reward_mean": float(np.mean(episode_rewards)), "episode_len_mean": float(np.mean(episode_lengths))})

    return policy, history
