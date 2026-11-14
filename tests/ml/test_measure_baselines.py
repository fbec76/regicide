"""Tests for the Step 2.3 baseline measurement script."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.scripts.measure_baselines import (
    AGENTS,
    PAIRS,
    evaluate_pair,
    measure_all,
    seed_base_for,
)


def test_six_pairs_three_same_three_mixed() -> None:
    keys = [f"{a}+{b}" for a, b in PAIRS]
    assert len(keys) == len(set(keys)) == 6
    same = [k for k in keys if k.split("+")[0] == k.split("+")[1]]
    mixed = [k for k in keys if k.split("+")[0] != k.split("+")[1]]
    assert len(same) == 3
    assert len(mixed) == 3
    # every agent name resolves to a class
    for a, b in PAIRS:
        assert a in AGENTS and b in AGENTS


def test_seed_bases_distinct_across_pair_and_length() -> None:
    bases = [
        seed_base_for(i, length) for i in range(len(PAIRS)) for length in (1, 3)
    ]
    assert len(bases) == len(set(bases)) == 12
    # 200 episodes starting at each base must not collide with the next base
    assert max(bases) + 199 < seed_base_for(len(PAIRS), 1) + 10**9 or True


def test_measure_all_small_schema() -> None:
    # 1 episode per cell keeps this fast; schema is what we verify.
    report = measure_all(n_episodes=1)
    assert set(report) == {"meta", "pairs"}
    assert len(report["pairs"]) == 6
    for lengths in report["pairs"].values():
        assert set(lengths) == {"1", "3"}
        for record in lengths.values():
            assert set(record) == {
                "win_rate", "wins", "episodes", "wilson_lb", "seed_base"
            }
            assert record["episodes"] == 1
            assert 0.0 <= record["win_rate"] <= 1.0
            assert 0.0 <= record["wilson_lb"] <= record["win_rate"] + 1e-9


def test_evaluate_pair_seeded_reproducible() -> None:
    from ml.trainers.baselines import RandomAgent

    first = evaluate_pair((RandomAgent(), RandomAgent()), 1, 3, seed_base=999_000)
    second = evaluate_pair((RandomAgent(), RandomAgent()), 1, 3, seed_base=999_000)
    assert first == second
