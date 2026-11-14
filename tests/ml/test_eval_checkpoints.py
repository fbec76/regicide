"""Tests for the Step 2.3 checkpoint evaluation script."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.scripts.eval_checkpoints import _STEP_RE, evaluate_run


def test_step_filename_regex_extracts_timesteps() -> None:
    match = _STEP_RE.search("ppo_model_100000_steps.zip")
    assert match is not None
    assert match.group(1) == "100000"
    assert _STEP_RE.search("best_model.zip") is None


def test_evaluate_run_missing_artifacts(tmp_path) -> None:
    # Empty run dir: no checkpoints, no final/best -> empty results list.
    report = evaluate_run(tmp_path, campaign_length=1, n_episodes=1, seed_base=0)
    assert report["results"] == []
    assert report["meta"]["campaign_length"] == 1


def test_evaluate_run_end_to_end_tiny(tmp_path) -> None:
    """400-step train on L=1 -> save/load round trip -> 2-episode eval."""
    from ml.trainers.ppo_trainer import TrainingConfig, train

    config = TrainingConfig(
        artifact_dir=str(tmp_path),
        campaign_length_schedule=[1],
        steps=400,
        n_envs=2,
        seed=0,
        eval_interval=400,
        n_eval_episodes=2,
    )
    result = train(config)
    run_dir = Path(result["run_dir"])

    report = evaluate_run(run_dir, campaign_length=1, n_episodes=2, seed_base=777_000)
    labels = [entry["model"] for entry in report["results"]]
    assert "final_model" in labels and "best_model" in labels
    for entry in report["results"]:
        assert entry["episodes"] == 2
        assert 0.0 <= entry["win_rate"] <= 1.0
        assert 0.0 <= entry["wilson_lb"] <= 1.0
        assert Path(entry["path"]).is_file()


def test_main_missing_run_dir_exits() -> None:
    from ml.scripts.eval_checkpoints import main

    sys_argv = sys.argv
    sys.argv = ["eval_checkpoints.py", "--run", "/nonexistent-run-dir-xyz"]
    try:
        with pytest.raises(SystemExit):
            main()
    finally:
        sys.argv = sys_argv
