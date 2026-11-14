"""Tests for the Step 4.2 eval harness (plan: test_eval.py).

Named tests from the plan:
  test_eval_win_rate_per_rung
  test_behavior_stats_damage_per_turn_by_position
  test_shield_timing_stat
  test_yield_rate_and_discard_efficiency
  test_round_robin_matrix_shape_and_symmetry
  test_eval_matrix_json_persisted

Runtime-bounded: one 2k-step trained model + two 400-step models (module
fixture); behavior-stat tests use synthetic step records (no training).
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.scripts.eval import (
    aggregate_behavior_stats,
    default_eval_rungs,
    discover_schedule,
    evaluate_matrix,
    evaluate_rung,
    gate_verdict,
    load_model,
    model_label,
    pair_key,
    run_roundrobin,
)


# --------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------- #

def _step(
    position=0,
    damage=0,
    shield=0,
    hp_fraction=1.0,
    yield_=False,
    suits=None,
    overspend=None,
) -> dict:
    return {
        "position": position,
        "damage_dealt": damage,
        "shield_raised": shield,
        "enemy_hp_fraction": hp_fraction,
        "yield_taken": yield_,
        "suit_powers": suits or {},
        "defense_overspend": overspend,
    }


def _episode(steps, win, hand_value=5) -> dict:
    return {"steps": steps, "win": win, "hand_value_end": hand_value}


def _train_tiny(tmp_path, steps, seed=0):
    from ml.trainers.ppo_trainer import TrainingConfig, train

    config = TrainingConfig(
        artifact_dir=str(tmp_path),
        campaign_length_schedule=[1],
        steps=steps,
        n_envs=2,
        seed=seed,
        eval_interval=steps,
        n_eval_episodes=2,
    )
    result = train(config)
    return Path(result["run_dir"])


@pytest.fixture(scope="module")
def two_tiny_models(tmp_path_factory):
    """Two distinct tiny runs for the round-robin pool."""
    run_dirs = []
    for seed in (0, 1):
        base = tmp_path_factory.mktemp(f"rr_seed{seed}")
        run_dirs.append(_train_tiny(base, steps=400, seed=seed))
    return run_dirs


# --------------------------------------------------------------------- #
# Win rate per rung (tiny trained model)
# --------------------------------------------------------------------- #

def test_eval_win_rate_per_rung(tmp_path) -> None:
    run_dir = _train_tiny(tmp_path, steps=2000, seed=0)

    # Rung discovery falls back to the run-name convention (no run_result.json
    # for direct train.py runs): <algo>_L1_s0 -> schedule [1] -> rungs [1, 12].
    assert discover_schedule(run_dir) == [1]
    assert default_eval_rungs(discover_schedule(run_dir)) == [1, 12]
    assert default_eval_rungs(None) == [12]

    model, model_path = load_model(run_dir)
    record = evaluate_rung(model, 1, n_episodes=3, seed_base=2_000_000)
    assert model_path.name in ("final_model.zip", "best_model.zip")

    assert set(record) >= {"win_rate", "wins", "episodes", "wilson_lb", "behavior"}
    assert record["episodes"] == 3
    assert 0.0 <= record["win_rate"] <= 1.0
    assert 0.0 <= record["wilson_lb"] <= record["win_rate"] + 1e-9
    behavior = record["behavior"]
    assert set(behavior) == {"by_position", "episode"}
    assert set(behavior["by_position"]) == {"0"}  # L=1 -> only position 0
    assert behavior["by_position"]["0"]["steps"] > 0
    assert behavior["episode"]["episodes"] == 3


def test_gate_verdict_phase2_pattern() -> None:
    # Strictly above AND Wilson LB above the baseline point estimate.
    assert gate_verdict(0.55, 0.45, 0.40) == "PPO_EXCEEDS_BASELINE"
    assert gate_verdict(0.55, 0.35, 0.40) == "PPO_BELOW_BASELINE"
    assert gate_verdict(0.40, 0.45, 0.40) == "PPO_BELOW_BASELINE"


# --------------------------------------------------------------------- #
# Behavior stats (synthetic records)
# --------------------------------------------------------------------- #

def test_behavior_stats_damage_per_turn_by_position() -> None:
    episodes = [
        _episode(
            [
                _step(position=0, damage=3),
                _step(position=0, damage=4),
                _step(position=1, damage=0),
                _step(position=1, damage=0),
            ],
            win=True,
        ),
        _episode([_step(position=0, damage=2)], win=False),
    ]
    stats = aggregate_behavior_stats(episodes)
    by_pos = stats["by_position"]
    assert set(by_pos) == {"0", "1"}  # sorted string position keys
    assert by_pos["0"]["steps"] == 3
    assert by_pos["0"]["damage_per_turn"] == pytest.approx(9.0 / 3.0)
    assert by_pos["1"]["steps"] == 2
    assert by_pos["1"]["damage_per_turn"] == pytest.approx(0.0)
    assert stats["episode"]["episodes"] == 2
    assert stats["episode"]["wins"] == 1


def test_shield_timing_stat() -> None:
    # Shields raised at hp fraction 1.0 and 0.5 -> mean 0.75; rate 2/3.
    episodes = [
        _episode(
            [
                _step(shield=1, hp_fraction=1.0),
                _step(shield=1, hp_fraction=0.5),
                _step(shield=0, hp_fraction=0.3),
            ],
            win=True,
        )
    ]
    pos = aggregate_behavior_stats(episodes)["by_position"]["0"]
    assert pos["shield_rate"] == pytest.approx(2.0 / 3.0)
    assert pos["shield_hp_fraction_mean"] == pytest.approx(0.75)

    # No shields at all -> None timing, zero rate.
    episodes = [_episode([_step(), _step()], win=False)]
    pos = aggregate_behavior_stats(episodes)["by_position"]["0"]
    assert pos["shield_rate"] == 0.0
    assert pos["shield_hp_fraction_mean"] is None


def test_yield_rate_and_discard_efficiency() -> None:
    episodes = [
        _episode(
            [
                _step(yield_=True, overspend=0),
                _step(yield_=False, overspend=2),
                _step(yield_=False, overspend=-1),
                _step(yield_=False),  # attack step: no defense overspend
            ],
            win=True,
            hand_value=7,
        ),
        _episode([_step(overspend=2)], win=False, hand_value=3),
    ]
    stats = aggregate_behavior_stats(episodes)
    pos = stats["by_position"]["0"]
    assert pos["yield_rate"] == pytest.approx(1.0 / 5.0)
    assert pos["defense_steps"] == 4
    assert pos["defense_overspend_mean"] == pytest.approx(3.0 / 4.0)
    assert pos["defense_overspend_dist"] == {"-1": 1, "0": 1, "2": 2}
    episode = stats["episode"]
    assert episode["hand_value_end_mean"] == pytest.approx(5.0)
    assert episode["hand_value_end_win_mean"] == pytest.approx(7.0)


def test_suit_power_usage_counts() -> None:
    episodes = [
        _episode(
            [
                _step(suits={"spades": "fires"}),
                _step(suits={"spades": "fires", "hearts": "pending"}),
                _step(suits={"diamonds": "blocked"}),
            ],
            win=True,
        )
    ]
    pos = aggregate_behavior_stats(episodes)["by_position"]["0"]
    assert pos["suit_powers"]["spades"] == {
        "fires": 2,
        "blocked": 0,
        "pending": 0,
    }
    assert pos["suit_powers"]["hearts"]["pending"] == 1
    assert pos["suit_powers"]["diamonds"]["blocked"] == 1


# --------------------------------------------------------------------- #
# Round-robin matrix
# --------------------------------------------------------------------- #

def test_round_robin_matrix_shape_and_symmetry(two_tiny_models) -> None:
    run_a, run_b = two_tiny_models
    models = [(model_label(run_dir), load_model(run_dir)[0]) for run_dir in (run_a, run_b)]
    matrix = evaluate_matrix(models, campaign_length=1, n_episodes=1, seed_base=3_000_000)

    name_a, name_b = matrix["models"]
    assert len(matrix["models"]) == 2
    # Pool of 2 -> 3 unordered pairs incl. the diagonal self-pairs.
    assert set(matrix["pairs"]) == {
        pair_key(name_a, name_a),
        pair_key(name_a, name_b),
        pair_key(name_b, name_b),
    }
    for pair in matrix["pairs"].values():
        assert 0.0 <= pair["joint_win_rate"] <= 1.0
        for rates in pair["seatings"].values():
            assert rates["episodes"] == 1
            assert set(rates) == {"win_rate", "wins", "episodes"}
    # Symmetry: either name order resolves to the same entry; off-diagonal
    # pairs average both seatings (recorded explicitly).
    off_diag = matrix["pairs"][pair_key(name_a, name_b)]
    assert set(off_diag["seatings"]) == {
        f"{name_a}@A+{name_b}@B",
        f"{name_b}@A+{name_a}@B",
    }
    assert matrix["pairs"][pair_key(name_b, name_a)] is off_diag
    assert matrix["meta"]["shape"]


def test_eval_matrix_json_persisted(tmp_path, two_tiny_models) -> None:
    out_path = tmp_path / "eval_matrix.json"
    matrix = run_roundrobin(
        two_tiny_models,
        campaign_length=1,
        n_episodes=1,
        seed_base=3_100_000,
        out_path=out_path,
    )
    assert out_path.is_file()
    loaded = json.loads(out_path.read_text())
    assert loaded["meta"]["campaign_length"] == 1
    assert loaded["meta"]["episodes_per_seating"] == 1
    assert loaded["models"] == matrix["models"]
    assert set(loaded) == {"meta", "models", "pairs"}
    for key, pair in loaded["pairs"].items():
        first, second = key.split("+")
        assert pair_key(first, second) == key  # canonical sorted key
        assert 0.0 <= pair["joint_win_rate"] <= 1.0
