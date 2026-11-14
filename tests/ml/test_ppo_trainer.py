"""Tests for Step 2.2 — MaskablePPO trainer smoke + artifact + mask-contract checks.

All runs use tmp_path artifact dirs and tiny budgets; the 2k-step smoke is
the plan's named gate (must stay well under ~60s).
"""
import os

import pytest

from ml.trainers.ppo_trainer import TrainingConfig, train

SMOKE_CONFIG_KWARGS = dict(
    campaign_length_schedule=[1],
    steps=2000,
    n_envs=2,
    seed=0,
    eval_interval=500,
    n_eval_episodes=5,
)


def test_short_training_smoke_2k_steps_runs(tmp_path) -> None:
    config = TrainingConfig(artifact_dir=str(tmp_path), **SMOKE_CONFIG_KWARGS)
    result = train(config)

    assert os.path.isdir(result["run_dir"])
    assert os.path.isfile(result["final_model"])
    assert os.path.isfile(result["best_model"])
    assert result["timesteps"] >= 2000  # SB3 overshoots to complete the final n_steps rollout chunk
    assert result["campaign_length_schedule"] == [1]


def test_no_invalid_action_errors_during_rollout(tmp_path) -> None:
    config = TrainingConfig(
        artifact_dir=str(tmp_path),
        campaign_length_schedule=[1],
        steps=800,
        n_envs=2,
        seed=0,
        eval_interval=400,
        n_eval_episodes=2,
    )
    result = train(config)
    # The mask contract: rollouts consume only masked (legal) actions — the
    # env's {"error": "Invalid action"} path must never be hit.
    assert result["invalid_action_count"] == 0


def test_checkpoint_and_monitor_csv_saved(tmp_path) -> None:
    config = TrainingConfig(
        artifact_dir=str(tmp_path),
        campaign_length_schedule=[1],
        steps=400,
        n_envs=2,
        seed=0,
        eval_interval=100,
        n_eval_episodes=2,
    )
    result = train(config)

    checkpoints = os.listdir(result["checkpoints_dir"])
    assert checkpoints, "no checkpoint files saved"
    assert all(name.endswith(".zip") for name in checkpoints)

    run_files = os.listdir(result["run_dir"])
    monitor_csvs = [name for name in run_files if name.startswith("monitor") and name.endswith(".csv")]
    assert monitor_csvs, f"no monitor CSV in {run_files}"
    assert "best_model.zip" in run_files
    assert "final_model.zip" in run_files


def test_trainer_rejects_unknown_algo() -> None:
    with pytest.raises(ValueError):
        train(TrainingConfig(algo="dqn", artifact_dir="/tmp/should-not-exist"))


def test_curriculum_callback_switches_rungs() -> None:
    from ml.trainers.ppo_trainer import _CurriculumCallback, _rung_boundaries

    # steps=1000, schedule [1, 3] -> single boundary at 500 timesteps.
    assert _rung_boundaries([1, 3], 1000) == [500]

    class _StubEnv:
        def __init__(self) -> None:
            class _Cfg:
                campaign_length = 1
            self.unwrapped = self
            self.config = _Cfg()

    stubs = [_StubEnv() for _ in range(2)]

    class _StubVecEnv:
        envs = stubs

    class _StubModel:
        def get_env(self):
            return _StubVecEnv()

    callback = _CurriculumCallback([1, 3], [500])
    callback.init_callback(_StubModel())
    callback.num_timesteps = 499
    assert callback._on_step() is True
    assert stubs[0].config.campaign_length == 1  # boundary not yet crossed

    callback.num_timesteps = 500
    assert callback._on_step() is True
    for stub in stubs:
        assert stub.config.campaign_length == 3  # applies at next episode reset
