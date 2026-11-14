"""Tests for Step 4.1 — RunConfig, the 5 starter configs, and the sweep runner."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml.configs.base import RunConfig
from ml.configs.sweep_starter import (
    STARTER_CONFIGS,
    STARTER_SEEDS,
    STARTER_SWEEP_ID,
)
from ml.scripts import sweep
from ml.trainers.ppo_trainer import TrainingConfig


def test_starter_configs_match_spec() -> None:
    """The 5 starter configs exactly per plan (reward / curriculum / device)."""
    assert len(STARTER_CONFIGS) == 5
    assert list(STARTER_CONFIGS) == [
        "aggressive_shaping_ladder_1_3",
        "minimal_shaping_ladder_1_3_6",
        "sparse_curriculum_ladder_1_3_6_9_12",
        "aggressive_shaping_fixed_L12",
        "aggressive_shaping_ladder_1_3_mps",
    ]

    c1 = STARTER_CONFIGS["aggressive_shaping_ladder_1_3"]
    assert c1.reward_config_name == "aggressive_shaping"
    assert c1.curriculum == [1, 3]
    assert c1.device == "cpu"

    c2 = STARTER_CONFIGS["minimal_shaping_ladder_1_3_6"]
    assert c2.reward_config_name == "minimal_shaping"
    assert c2.curriculum == [1, 3, 6]
    assert c2.device == "cpu"

    c3 = STARTER_CONFIGS["sparse_curriculum_ladder_1_3_6_9_12"]
    assert c3.reward_config_name == "sparse_curriculum"
    assert c3.curriculum == [1, 3, 6, 9, 12]
    assert c3.device == "cpu"

    c4 = STARTER_CONFIGS["aggressive_shaping_fixed_L12"]
    assert c4.reward_config_name == "aggressive_shaping"
    assert c4.curriculum is None  # fixed-length ablation control
    assert c4.campaign_length == 12
    assert c4.device == "cpu"

    c5 = STARTER_CONFIGS["aggressive_shaping_ladder_1_3_mps"]
    assert c5.reward_config_name == "aggressive_shaping"
    assert c5.curriculum == [1, 3]
    assert c5.device == "mps"

    for config in STARTER_CONFIGS.values():
        assert config.algo == "ppo"
        assert config.steps == 1_000_000
        assert config.telemetry is True
        # M3: gamma/ent_coef sweep dims are follow-up configs — starters use
        # the Phase 2 defaults.
        assert config.gamma == 0.99
        assert config.ent_coef == 0.01


def test_sweep_dry_run_enumerates_15_runs(capsys: pytest.CaptureFixture) -> None:
    """3 seeds x 5 configs = 15 runs with seed<k> dirs, dry-run lists all 15."""
    runs = sweep.enumerate_runs(STARTER_CONFIGS, STARTER_SEEDS, STARTER_SWEEP_ID)
    assert len(runs) == 15
    assert len({(run.config_name, run.seed) for run in runs}) == 15

    names = {run.config_name for run in runs}
    assert names == set(STARTER_CONFIGS)
    for name in STARTER_CONFIGS:
        seeds = {run.seed for run in runs if run.config_name == name}
        assert seeds == {0, 1, 2}

    expected_dirs = {
        sweep.run_dir_for(sweep.SWEEP_ROOT, STARTER_SWEEP_ID, name, seed)
        for name in STARTER_CONFIGS
        for seed in STARTER_SEEDS
    }
    assert {run.run_dir for run in runs} == {str(d) for d in expected_dirs}
    for run in runs:
        assert f"seed{run.seed}" in run.run_dir
        assert run.config_name in run.run_dir

    sweep._print_dry_run(runs)
    out = capsys.readouterr().out
    assert "15 runs total" in out
    assert sum(1 for line in out.splitlines() if line.startswith("[")) == 15


def test_sweep_resume_skips_completed(tmp_path) -> None:
    """A seed dir with run_result.json is skipped; others run; manifest records both."""
    runs = sweep.enumerate_runs(
        STARTER_CONFIGS, STARTER_SEEDS, STARTER_SWEEP_ID, sweep_root=tmp_path
    )
    first = runs[0]
    completed_dir = sweep.run_dir_for(
        tmp_path, STARTER_SWEEP_ID, first.config_name, first.seed
    )
    completed_dir.mkdir(parents=True)
    (completed_dir / sweep.RUN_RESULT_FILE).write_text(json.dumps({"timesteps": 1}))

    calls = []

    def stub_train(run) -> dict:
        calls.append((run.config_name, run.seed))
        return {"run_dir": run.run_dir, "timesteps": 1234}

    summary = sweep.execute_sweep(
        runs, STARTER_SWEEP_ID, sweep_root=tmp_path, train_fn=stub_train
    )

    assert calls == [(run.config_name, run.seed) for run in runs[1:]]
    assert summary["skipped"] == 1
    assert summary["completed"] == 14
    assert summary["failed"] == 0

    # Completed dir untouched, executed dirs got their own result files.
    assert json.loads((completed_dir / sweep.RUN_RESULT_FILE).read_text())["timesteps"] == 1
    for run in runs[1:]:
        assert (Path(run.run_dir) / sweep.RUN_RESULT_FILE).is_file()

    manifest = json.loads((tmp_path / STARTER_SWEEP_ID / sweep.MANIFEST_FILE).read_text())
    assert manifest["sweep_id"] == STARTER_SWEEP_ID
    assert len(manifest["runs"]) == 15
    statuses = {entry["run_dir"]: entry["status"] for entry in manifest["runs"]}
    assert statuses[str(completed_dir)] == "completed"
    assert statuses[str(runs[1].run_dir)] == "completed"
    assert all(
        status == "completed"
        for run_dir, status in statuses.items()
        if run_dir != str(completed_dir)
    )

    # Re-invocation is a no-op: everything skipped.
    calls.clear()
    summary2 = sweep.execute_sweep(
        runs, STARTER_SWEEP_ID, sweep_root=tmp_path, train_fn=stub_train
    )
    assert calls == []
    assert summary2["skipped"] == 15
    assert summary2["completed"] == 0


def test_run_config_maps_to_training_config() -> None:
    """RunConfig fields land on TrainingConfig; ladder vs fixed schedule."""
    ladder = RunConfig(
        name="x",
        reward_config_name="minimal_shaping",
        curriculum=[1, 3],
        steps=500_000,
        n_envs=4,
        seed=7,
        device="mps",
        lr=1e-4,
        gamma=0.95,
        ent_coef=0.05,
        gae_lambda=0.8,
        n_steps=256,
        batch_size=128,
        eval_interval=10_000,
        n_eval_episodes=9,
        telemetry=False,
    )
    tc = ladder.to_training_config(artifact_dir="some/dir")
    assert isinstance(tc, TrainingConfig)
    assert tc.campaign_length_schedule == [1, 3]
    assert tc.reward_config_name == "minimal_shaping"
    assert tc.steps == 500_000
    assert tc.n_envs == 4
    assert tc.seed == 7
    assert tc.device == "mps"
    assert tc.lr == 1e-4
    assert tc.gamma == 0.95
    assert tc.ent_coef == 0.05
    assert tc.gae_lambda == 0.8
    assert tc.n_steps == 256
    assert tc.batch_size == 128
    assert tc.eval_interval == 10_000
    assert tc.n_eval_episodes == 9
    assert tc.telemetry is False
    assert tc.artifact_dir == "some/dir"

    fixed = RunConfig(name="y", curriculum=None, campaign_length=12)
    assert fixed.to_training_config("d").campaign_length_schedule == [12]
