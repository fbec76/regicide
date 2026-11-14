"""MaskablePPO trainer (Step 2.2). ALL stable-baselines3 / sb3-contrib usage
lives in ml/trainers/** (ADR-002 import boundary — enforced by
tests/ml/test_boundaries.py, which also forbids torch/sb3 in ml/envs/**).

Training setup (decisions.md authority, plan "Training setup"):
    DummyVecEnv(8) + ActionMasker + MaskablePPO("MlpPolicy"), n_envs 8,
    gamma 0.99, ent_coef 0.01, lr 2.5e-4 (linear), gae_lambda 0.9,
    n_steps 512 (rollout 512*8=4096), batch 64, clip_range 0.2 (linear),
    device cpu, tensorboard_log + Monitor CSV, MaskableEvalCallback
    (eval_freq is per-env steps), deterministic=True eval actions.

Curriculum (campaign_length schedule, e.g. [1, 3] for the proof run)
-------------------------------------------------------------------
Simplest correct approach: ONE DummyVecEnv + ONE learn() call; a callback
switches each env's ``config.campaign_length`` to the next rung when the
global timestep count crosses the (equal-weight) rung boundary. Because the
env reads campaign_length only in reset(), a switch takes effect at the next
episode boundary — never mid-episode. Keeping a single learn() call also
keeps the linear lr/clip schedules aligned with the FULL run length.

Determinism (M5, run-level): ``seed_everything`` seeds random/numpy/torch ONCE
before env construction; DummyVecEnv steps envs sequentially so global-random
consumption stays atomic. Training is run-level deterministic, NOT per-env
reproducible (the engine shuffles read module-level ``random``).
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker

from ml.envs.regicide_env import EnvConfig
from ml.trainers.common import make_env, seed_everything

logger = logging.getLogger(__name__)

DEFAULT_CLIP_RANGE = 0.2
DEFAULT_EVAL_EPISODES = 20
ARTIFACT_DIR_DEFAULT = "ml/runs"


def _action_masks_fn(env: gym.Env) -> np.ndarray:
    """ActionMasker callback: the Regicide env exposes the mask natively."""
    return env.action_masks()


def _linear_decay(initial_value: float) -> LinearSchedule:
    """RL-Zoo-style linear schedule: value decays to ~0 over the full run."""
    return LinearSchedule(initial_value, 0.0, 1.0)


@dataclass
class TrainingConfig:
    """PPO training configuration (Step 2.2 contract).

    gamma MUST equal the env's PBRS gamma (passed to EnvConfig at env
    construction). campaign_length_schedule is the curriculum ladder, e.g.
    [1, 3]; a single-entry list trains at a fixed campaign length. Rung
    boundaries split ``steps`` equally across rungs.
    """

    algo: str = "ppo"
    campaign_length_schedule: list = field(default_factory=lambda: [1, 3])
    steps: int = 1_000_000
    n_envs: int = 8
    seed: int = 0
    device: str = "cpu"
    lr: float = 2.5e-4
    gamma: float = 0.99
    ent_coef: float = 0.01
    gae_lambda: float = 0.9
    n_steps: int = 512
    batch_size: int = 64
    eval_interval: int = 50_000
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES
    artifact_dir: str = ARTIFACT_DIR_DEFAULT


class InvalidActionGuard(gym.Wrapper):
    """Rollout instrumentation: counts steps whose info carries an error
    (the env's ``{"error": "Invalid action"}`` path must be unreachable when
    actions are selected under the mask)."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.invalid_action_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if "error" in info:
            self.invalid_action_count += 1
        return obs, reward, terminated, truncated, info


def count_invalid_actions(vec_env) -> int:
    """Sum InvalidActionGuard counters across a DummyVecEnv's wrapped envs."""
    total = 0
    for wrapped in vec_env.envs:
        node = wrapped
        while node is not None:
            if isinstance(node, InvalidActionGuard):
                total += node.invalid_action_count
                break
            node = getattr(node, "env", None)
    return total


class _CurriculumCallback(BaseCallback):
    """Switch campaign_length rungs at equal-weight timestep boundaries.

    The switch mutates each env's config; it applies at the next episode
    reset (campaign_length is only read in RegicideGymEnv.reset).
    """

    def __init__(self, schedule: list, boundaries: list, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.schedule = schedule
        self.boundaries = boundaries  # cumulative num_timesteps at which rung k ends
        self.rung = 0

    def _on_step(self) -> bool:
        while self.rung < len(self.boundaries) and self.num_timesteps >= self.boundaries[self.rung]:
            self.rung += 1
            new_length = self.schedule[self.rung]
            for wrapped in self.training_env.envs:
                wrapped.unwrapped.config.campaign_length = new_length
            if self.verbose:
                logger.info("Curriculum: campaign_length -> %d at step %d", new_length, self.num_timesteps)
        return True


def _rung_boundaries(schedule: list, steps: int) -> list:
    """Equal-weight split: cumulative timestep counts ending each rung."""
    per_rung = steps // len(schedule)
    return [per_rung * (k + 1) for k in range(len(schedule) - 1)]


def _make_run_dir(config: TrainingConfig) -> Path:
    run_name = "{}_L{}_s{}".format(
        config.algo,
        "-".join(str(length) for length in config.campaign_length_schedule),
        config.seed,
    )
    run_dir = Path(config.artifact_dir) / run_name / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _train_env_fn(
    config: TrainingConfig, campaign_length: int, run_dir: Path, rank: int
) -> Callable[[], gym.Env]:
    def _make() -> gym.Env:
        env = make_env(
            EnvConfig(campaign_length=campaign_length, gamma=config.gamma),
            seed=config.seed,
            rank=rank,
        )
        env = ActionMasker(env, _action_masks_fn)
        env = InvalidActionGuard(env)
        return Monitor(
            env, filename=str(run_dir / f"monitor_{rank}"), info_keywords=("result",)
        )

    return _make


def _make_eval_env(config: TrainingConfig, campaign_length: int, run_dir: Path) -> gym.Env:
    env = make_env(
        EnvConfig(campaign_length=campaign_length, gamma=config.gamma),
        seed=config.seed,
        rank=config.n_envs,  # distinct traceability seed; eval resets re-seed anyway
    )
    env = ActionMasker(env, _action_masks_fn)
    env = InvalidActionGuard(env)
    return Monitor(
        env, filename=str(run_dir / "monitor_eval"), info_keywords=("result",)
    )


def train(config: TrainingConfig) -> dict:
    """Run one MaskablePPO training job; returns artifact paths + rollout audit."""
    if config.algo != "ppo":
        raise ValueError(f"ppo_trainer supports algo='ppo', got {config.algo!r}")
    schedule = config.campaign_length_schedule
    if not schedule or not all(isinstance(l, int) and l >= 1 for l in schedule):
        raise ValueError(f"campaign_length_schedule must be positive ints, got {schedule!r}")
    if config.steps < 1 or config.n_envs < 1:
        raise ValueError("steps and n_envs must be positive")

    seed_everything(config.seed)  # single run-level seed, BEFORE env construction
    run_dir = _make_run_dir(config)

    train_env = DummyVecEnv(
        [_train_env_fn(config, schedule[0], run_dir, rank) for rank in range(config.n_envs)]
    )
    eval_env = _make_eval_env(config, schedule[-1], run_dir)

    curriculum_cb = _CurriculumCallback(schedule, _rung_boundaries(schedule, config.steps))
    eval_cb = MaskableEvalCallback(
        eval_env,
        n_eval_episodes=config.n_eval_episodes,
        eval_freq=config.eval_interval,  # per-env steps (plan: eval_freq is per-env)
        deterministic=True,  # M5 eval convention
        best_model_save_path=str(run_dir),
        verbose=1,
    )
    checkpoint_cb = CheckpointCallback(
        save_freq=config.eval_interval,  # per-env steps
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_model",
        verbose=1,
    )

    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=_linear_decay(config.lr),
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        ent_coef=config.ent_coef,
        clip_range=_linear_decay(DEFAULT_CLIP_RANGE),
        seed=config.seed,
        device=config.device,
        verbose=1,
        tensorboard_log=str(run_dir),
    )

    model.learn(
        config.steps,
        callback=[curriculum_cb, eval_cb, checkpoint_cb],
        reset_num_timesteps=True,
    )

    final_model_path = run_dir / "final_model.zip"
    model.save(str(final_model_path.with_suffix("")))
    train_env.close()
    eval_env.close()

    invalid = count_invalid_actions(train_env)
    if invalid:
        warnings.warn(
            f"{invalid} invalid-action responses during rollout — the mask "
            "contract was violated (should be 0)"
        )

    return {
        "run_dir": str(run_dir),
        "final_model": str(final_model_path),
        "best_model": str(run_dir / "best_model.zip"),
        "checkpoints_dir": str(run_dir / "checkpoints"),
        "timesteps": int(model.num_timesteps),
        "invalid_action_count": invalid,
        "campaign_length_schedule": list(schedule),
    }


__all__ = [
    "TrainingConfig",
    "InvalidActionGuard",
    "count_invalid_actions",
    "train",
]
