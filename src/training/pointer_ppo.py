"""PPO for the pointer agent: batched rollouts on PointerEnv, GAE, minibatch clipped updates.

Rollouts store only picked token indices; a step's observation is rebuilt from (scan, order, t,
belief) at update time, so memory stays small even with ~800 candidates per scan.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from env.pointer_env import PointerEnv, build_obs


@dataclass
class PointerPPOConfig:
    episodes_per_update: int = 128
    epochs: int = 4
    minibatch_size: int = 512
    gamma: float = 1.0          # 1 keeps the return equal to the reader's total loss drop
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    lr: float = 3e-4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5


@torch.no_grad()
def rollout(env: PointerEnv, policy: nn.Module, scans: torch.Tensor, groups: torch.Tensor, greedy: bool = False) -> dict[str, torch.Tensor]:
    env.reset(scans, groups)
    n, steps, dev = len(env.scans), env.budget, env.bank.device
    buf = {name: torch.zeros(n, steps, device=dev) for name in ("logp", "value", "reward", "belief")}
    buf.update(active=torch.zeros(n, steps, dtype=torch.bool, device=dev), stop=torch.zeros(n, steps, dtype=torch.bool, device=dev),
               token=torch.zeros(n, steps, dtype=torch.long, device=dev))
    was_training = policy.training
    policy.eval()
    for s in range(steps):
        if bool(env.done.all()):
            break
        active = ~env.done
        logits, value = policy(env.observation())
        dist = torch.distributions.Categorical(logits=logits)
        n_cand = logits.shape[1] - 1
        if greedy:
            # STOP competes with ~N tokens, so a plain argmax almost always stops. Decide in two parts
            # (same distribution): stop if P(STOP) > 1/2, otherwise the most likely token.
            p_stop = dist.probs[:, n_cand]
            action = torch.where(p_stop > 0.5, n_cand, logits[:, :n_cand].argmax(-1))
        else:
            action = dist.sample()
        stop = action == n_cand
        token = action.clamp(max=n_cand - 1)
        buf["belief"][:, s] = env.belief
        buf["logp"][:, s] = dist.log_prob(action)
        buf["value"][:, s] = value
        buf["active"][:, s] = active
        buf["stop"][:, s] = stop & active
        buf["token"][:, s] = token
        buf["reward"][:, s] = env.step(stop, token)
    policy.train(was_training)
    buf.update(scans=env.scans, groups=env.groups, labels=env.labels, order=env.order.clone(), t=env.t.clone(), p_final=env.prediction())
    return buf


def gae(buf: dict[str, torch.Tensor], gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns [B, S]; a step is terminal when the next step is inactive."""
    value, reward, active = buf["value"], buf["reward"], buf["active"].float()
    next_active = torch.cat([active[:, 1:], torch.zeros_like(active[:, :1])], dim=1)
    next_value = torch.cat([value[:, 1:], torch.zeros_like(value[:, :1])], dim=1) * next_active
    delta = (reward + gamma * next_value - value) * active
    adv = torch.zeros_like(value)
    running = torch.zeros_like(value[:, 0])
    for s in reversed(range(value.shape[1])):
        running = delta[:, s] + gamma * lam * next_active[:, s] * running
        adv[:, s] = running
    adv = adv * active
    return adv, adv + value


def ppo_update(policy: nn.Module, optimizer: torch.optim.Optimizer, env: PointerEnv, buf: dict[str, torch.Tensor], cfg: PointerPPOConfig) -> dict[str, float]:
    adv, ret = gae(buf, cfg.gamma, cfg.gae_lambda)
    b, s = buf["active"].nonzero(as_tuple=True)
    adv_f, ret_f = adv[b, s], ret[b, s]
    adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
    old_logp = buf["logp"][b, s]
    stats: dict[str, list[float]] = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": []}
    policy.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(len(b), device=b.device)
        for start in range(0, len(perm), cfg.minibatch_size):
            mb = perm[start:start + cfg.minibatch_size]
            rows, steps = b[mb], s[mb]
            obs = build_obs(env.bank, buf["scans"][rows], buf["order"][rows], steps, buf["belief"][rows, steps], env.budget)
            logits, value = policy(obs)
            n_cand = logits.shape[1] - 1
            action = torch.where(buf["stop"][rows, steps], n_cand, buf["token"][rows, steps])
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(action)
            ratio = (logp - old_logp[mb]).exp()
            a = adv_f[mb]
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip_ratio, 1 + cfg.clip_ratio) * a).mean()
            value_loss = nn.functional.mse_loss(value, ret_f[mb])
            entropy = dist.entropy().mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                stats["policy_loss"].append(float(policy_loss))
                stats["value_loss"].append(float(value_loss))
                stats["entropy"].append(float(entropy))
                stats["approx_kl"].append(float((old_logp[mb] - logp).mean()))
                stats["clip_frac"].append(float(((ratio - 1).abs() > cfg.clip_ratio).float().mean()))
    return {k: float(np.mean(v)) for k, v in stats.items()}
