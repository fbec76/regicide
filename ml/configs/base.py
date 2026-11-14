"""RunConfig (Step 4.1): sweep-level training configuration.

A RunConfig describes ONE named configuration; a sweep materializes one run
per (config, seed) pair. ``to_training_config`` maps it into the trainer's
TrainingConfig — the single place RunConfig fields meet the trainer contract.

Curriculum semantics: ``curriculum`` is the campaign_length ladder (e.g.
[1, 3]); ``None`` means fixed-length training at ``campaign_length``. The
gamma/ent_coef sweep axes (plan M3 / Sweep axis table) are follow-up configs
beyond the 5 starters — the defaults here match the Phase 2 training setup.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ml.trainers.ppo_trainer import TrainingConfig

DEFAULT_STEPS = 1_000_000
DEFAULT_N_ENVS = 8
DEFAULT_LR = 2.5e-4
DEFAULT_GAMMA = 0.99
DEFAULT_ENT_COEF = 0.01
DEFAULT_GAE_LAMBDA = 0.9
DEFAULT_N_STEPS = 512
DEFAULT_BATCH_SIZE = 64
DEFAULT_EVAL_INTERVAL = 50_000
DEFAULT_EVAL_EPISODES = 20


@dataclass
class RunConfig:
    """One named sweep configuration (seed is overridden per run)."""

    name: str
    algo: str = "ppo"
    reward_config_name: str = "aggressive_shaping"
    curriculum: Optional[List[int]] = None
    campaign_length: int = 12
    steps: int = DEFAULT_STEPS
    n_envs: int = DEFAULT_N_ENVS
    seed: int = 0
    device: str = "cpu"
    lr: float = DEFAULT_LR
    gamma: float = DEFAULT_GAMMA
    ent_coef: float = DEFAULT_ENT_COEF
    gae_lambda: float = DEFAULT_GAE_LAMBDA
    n_steps: int = DEFAULT_N_STEPS
    batch_size: int = DEFAULT_BATCH_SIZE
    eval_interval: int = DEFAULT_EVAL_INTERVAL
    n_eval_episodes: int = DEFAULT_EVAL_EPISODES
    telemetry: bool = True

    def campaign_length_schedule(self) -> List[int]:
        """Ladder if curriculum is set, else the single fixed length."""
        if self.curriculum is not None:
            return list(self.curriculum)
        return [self.campaign_length]

    def to_training_config(self, artifact_dir: str) -> TrainingConfig:
        return TrainingConfig(
            algo=self.algo,
            campaign_length_schedule=self.campaign_length_schedule(),
            steps=self.steps,
            n_envs=self.n_envs,
            seed=self.seed,
            device=self.device,
            lr=self.lr,
            gamma=self.gamma,
            ent_coef=self.ent_coef,
            gae_lambda=self.gae_lambda,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            eval_interval=self.eval_interval,
            n_eval_episodes=self.n_eval_episodes,
            reward_config_name=self.reward_config_name,
            telemetry=self.telemetry,
            artifact_dir=artifact_dir,
        )


@dataclass
class RunSpec:
    """One materialized (config, seed) run within a sweep."""

    config: RunConfig
    seed: int
    run_dir: str

    @property
    def config_name(self) -> str:
        return self.config.name


__all__ = ["RunConfig", "RunSpec"]
