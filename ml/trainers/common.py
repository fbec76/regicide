"""Shared trainer utilities (Step 2.1).

Seeding, env construction, and the evaluation harness used by every trainer
(PPO now, DQN in Phase 5). ALL torch/sb3 usage stays inside ml/trainers/**
(ADR-002): this module imports torch for seeding only and duck-types the
policy (any object exposing ``predict(obs, action_masks=..., deterministic=...)``
— MaskablePPO or a scripted-baseline adapter).

Determinism convention (M5, run-level)
--------------------------------------
The engine's deck shuffles read the GLOBAL ``random`` module, so training is
deterministic only at run level: one global seed set once per run (via
``seed_everything``) BEFORE env construction, and DummyVecEnv (sequential
stepping keeps global-random consumption atomic). Evaluation is per-episode
reproducible instead: a fresh env per episode with ``reset(seed=base+i)``
(a provided seed re-seeds the global RNG per the Gymnasium/M5 convention).
"""
from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Optional

import numpy as np
import torch

from ml.envs.action_mask import N_ACTIONS
from ml.envs.regicide_env import EnvConfig, RegicideGymEnv

DEFAULT_WILSON_Z = 1.96  # two-sided 95% confidence


def seed_everything(seed: int) -> None:
    """Seed the global RNG stack once per run (run-level determinism, M5).

    torch.manual_seed also seeds CUDA when available; this slice is CPU-only
    (device="cpu" is the trainer default).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_env(
    config: EnvConfig, seed: Optional[int] = None, rank: int = 0
) -> RegicideGymEnv:
    """Env factory: one RegicideGymEnv for ``config``.

    NO VecEnv here — wrapping (ActionMasker/Monitor/DummyVecEnv) is
    trainer-level. ``seed``/``rank`` only record an intended episode seed on
    the config (seed + rank) for traceability; actual seeding happens via
    ``env.reset(seed=...)`` at the call site (M5 convention).
    """
    if seed is not None:
        config = replace(config, seed=seed + rank)
    return RegicideGymEnv(config)


def evaluate_policy(
    model: Any,
    campaign_length: int,
    n_episodes: int,
    seed_base: int,
    deterministic: bool = True,
) -> dict:
    """Win-rate evaluation: the policy plays BOTH seats of a single seeded env.

    One fresh RegicideGymEnv per episode, reset(seed=seed_base + i) (a
    provided seed re-seeds the global RNG — M5). Every step the active seat's
    obs is passed to ``model.predict(obs, action_masks=info["mask"], ...)``
    so the policy only ever selects unmasked (legal) actions. A win is a
    terminated episode whose engine result is "win" (truncation at the env
    step_limit is not a win). Deterministic=True per the M5 eval convention.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be >= 1, got {n_episodes}")
    wins = 0
    for i in range(n_episodes):
        seed = seed_base + i
        env = make_env(EnvConfig(campaign_length=campaign_length), seed=seed, rank=i)
        obs, info = env.reset(seed=seed)
        terminated = truncated = False
        while not (terminated or truncated):
            action, _ = model.predict(
                obs, action_masks=info["mask"], deterministic=deterministic
            )
            obs, _, terminated, truncated, info = env.step(int(action))
        if terminated and info.get("result") == "win":
            wins += 1
        env.close()
    return {
        "win_rate": wins / n_episodes,
        "wins": wins,
        "episodes": n_episodes,
        "campaign_length": campaign_length,
    }


def wilson_lower_bound(wins: int, n: int, z: float = DEFAULT_WILSON_Z) -> float:
    """Wilson score interval LOWER bound (95% by default) for a binomial win rate."""
    if n <= 0:
        raise ValueError(f"n must be a positive episode count, got {n}")
    if not 0 <= wins <= n:
        raise ValueError(f"wins must be in [0, n], got wins={wins}, n={n}")
    phat = wins / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = phat + z2 / (2.0 * n)
    margin = z * np.sqrt((phat * (1.0 - phat) + z2 / (4.0 * n)) / n)
    return float(max(0.0, min(1.0, (centre - margin) / denom)))


__all__ = [
    "N_ACTIONS",
    "seed_everything",
    "make_env",
    "evaluate_policy",
    "wilson_lower_bound",
]
