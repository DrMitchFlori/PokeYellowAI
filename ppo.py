# coding: utf-8
"""Reusable PPO components for PokeYellowAI.

This module exposes a simple actor-critic network, rollout gathering
utilities and a PPO update step. It also provides ``load_config`` for
reading hyperparameters from JSON or YAML files.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:  # Optional heavy deps
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
except ModuleNotFoundError:  # pragma: no cover - modules may be missing during tests
    np = None
    torch = None
    nn = None
    F = None
    optim = None

from poke_rewards import check_goals

Goal = Dict[str, object]


def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration dictionary from JSON or YAML."""
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(('.yml', '.yaml')):
            if yaml is None:
                raise ImportError('PyYAML is required for YAML config files')
            return yaml.safe_load(f)
        return json.load(f)


class Curriculum:
    """Manage goal progression during training."""

    def __init__(self, goals: Sequence[Goal], threshold: float = 0.8) -> None:
        self.goals = {g["id"]: g for g in goals}
        self.threshold = threshold
        self.stats: Dict[str, Dict[str, int]] = {
            g["id"]: {"successes": 0, "attempts": 0} for g in goals
        }
        self.active: set[str] = {g["id"] for g in goals if not g.get("prerequisites")}

    def active_goals(self) -> List[Goal]:
        return [self.goals[g] for g in self.active]

    def record_episode(self, triggered: Iterable[str]) -> None:
        triggered_set = set(triggered)
        for gid in list(self.active):
            self.stats[gid]["attempts"] += 1
            if gid in triggered_set:
                self.stats[gid]["successes"] += 1
        self._update_unlocks()

    def _update_unlocks(self) -> None:
        for gid, goal in self.goals.items():
            if gid in self.active:
                continue
            prereqs = goal.get("prerequisites", [])
            if not prereqs:
                continue
            if all(
                self.stats[p]["attempts"] > 0
                and self.stats[p]["successes"] / self.stats[p]["attempts"] >= self.threshold
                for p in prereqs
            ):
                self.active.add(gid)


if torch is not None and np is not None:
    class ActorCritic(nn.Module):
        """Simple convolutional actor-critic network."""

        def __init__(self, obs_shape: Sequence[int], n_actions: int) -> None:
            super().__init__()
            c, h, w = obs_shape
            self.features = nn.Sequential(
                nn.Conv2d(c, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                n_flatten = self.features(torch.zeros(1, *obs_shape)).shape[1]
            self.policy = nn.Sequential(
                nn.Linear(n_flatten, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions),
            )
            self.value = nn.Sequential(
                nn.Linear(n_flatten, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
            )

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            x = x / 255.0
            features = self.features(x)
            return self.policy(features), self.value(features)

        def act(self, x: np.ndarray) -> Tuple[int, float, float]:
            """Return action, log probability and value for a single observation."""
            x = x.transpose(2, 0, 1)
            with torch.no_grad():
                logits, value = self.forward(torch.from_numpy(x).float().unsqueeze(0))
                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            return action.item(), log_prob.item(), value.item()

        def evaluate(self, states: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            logits, values = self.forward(states)
            dist = torch.distributions.Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return log_probs, entropy, values.squeeze(-1)
else:  # pragma: no cover - used when dependencies missing
    class ActorCritic:  # type: ignore
        def __init__(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
            raise ImportError("PyTorch and NumPy are required to use ActorCritic")

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise ImportError("PyTorch is required to use ActorCritic")


def compute_gae(
    rewards: List[float],
    values: List[float],
    dones: List[bool],
    gamma: float = 0.99,
    lam: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ImportError("PyTorch is required for compute_gae")
    adv = 0.0
    advantages: List[float] = []
    last_value = 0.0
    for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
        delta = r + gamma * last_value * (1.0 - d) - v
        adv = delta + gamma * lam * (1.0 - d) * adv
        advantages.insert(0, adv)
        last_value = v
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    returns_t = advantages_t + torch.tensor(values, dtype=torch.float32)
    return advantages_t, returns_t


def gather_rollout(env: Any, model: ActorCritic, curriculum: Curriculum, rollout_steps: int) -> Dict[str, List]:
    if np is None or torch is None:
        raise ImportError("NumPy and PyTorch are required for gather_rollout")
    obs = env.reset()
    prev_mem = env.get_ram()
    storage = defaultdict(list)
    episode_goals: set[str] = set()

    for _ in range(rollout_steps):
        action, log_p, value = model.act(obs)
        next_obs, reward, done, _info = env.step(action)
        curr_mem = env.get_ram()
        triggered = check_goals(prev_mem, curr_mem, curriculum.active_goals())
        shaped = reward + sum(r for _g, r in triggered)
        episode_goals.update(g for g, _r in triggered)

        obs_t = obs.transpose(2, 0, 1)
        storage["states"].append(torch.from_numpy(obs_t).float())
        storage["actions"].append(action)
        storage["log_probs"].append(log_p)
        storage["values"].append(value)
        storage["rewards"].append(shaped)
        storage["dones"].append(done)

        if done:
            obs = env.reset()
            prev_mem = env.get_ram()
            curriculum.record_episode(episode_goals)
            episode_goals = set()
        else:
            obs = next_obs
            prev_mem = curr_mem

    return storage


def ppo_update(
    model: ActorCritic,
    optimizer: optim.Optimizer,
    rollout: Dict[str, List],
    *,
    clip_range: float = 0.2,
    epochs: int = 4,
    batch_size: int = 64,
    vf_coef: float = 0.5,
    ent_coef: float = 0.01,
    gamma: float = 0.99,
    lam: float = 0.95,
) -> None:
    if torch is None:
        raise ImportError("PyTorch is required for ppo_update")
    states = torch.stack(rollout["states"])
    actions = torch.tensor(rollout["actions"])
    old_log_probs = torch.tensor(rollout["log_probs"])
    values = rollout["values"]
    rewards = rollout["rewards"]
    dones = rollout["dones"]

    advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    dataset_size = states.size(0)
    for _ in range(epochs):
        indices = torch.randperm(dataset_size)
        for start in range(0, dataset_size, batch_size):
            idx = indices[start : start + batch_size]
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log = old_log_probs[idx]
            batch_adv = advantages[idx]
            batch_ret = returns[idx]

            log_probs, entropy, values_pred = model.evaluate(batch_states, batch_actions)
            ratio = torch.exp(log_probs - batch_old_log)
            obj = ratio * batch_adv
            clipped_obj = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * batch_adv
            policy_loss = -torch.min(obj, clipped_obj).mean()
            value_loss = F.mse_loss(values_pred, batch_ret)
            loss = policy_loss + vf_coef * value_loss - ent_coef * entropy.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


