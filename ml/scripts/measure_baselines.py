"""Measure scripted-baseline TEAM win rates (Step 2.3, deliverable 1).

All 6 unordered pair combos of {random, greedy_highest, shield_greedy} (3
same-agent pairs + 3 mixed), each playing campaign_length in {1, 3} for 200
episodes on seeded single envs. Both seats cooperate (Regicide 2p is a team
game): the first name in a pair key is seat A (Player1), the second seat B
(Player2). Output: ml/runs/baselines/baselines.json (gitignored — ml/runs/).

Seeding (M5): a fresh env per episode with reset(seed=seed_base + i); the
provided seed re-seeds the global ``random`` module (engine shuffles read it).
``np.random`` is additionally seeded with the same episode seed so
RandomAgent (which samples via np.random.choice) is per-episode reproducible
too. seed_base is distinct per pair x campaign length:
``seed_base = pair_index * 10_000 + campaign_length * 1_000``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from ml.envs.regicide_env import EnvConfig, RegicideGymEnv
from ml.trainers.baselines import (
    BaselineAgent,
    GreedyHighestAgent,
    RandomAgent,
    ShieldGreedyAgent,
)
from ml.trainers.common import wilson_lower_bound

EPISODES = 200
CAMPAIGN_LENGTHS = (1, 3)

AGENTS: dict = {
    "random": RandomAgent,
    "greedy_highest": GreedyHighestAgent,
    "shield_greedy": ShieldGreedyAgent,
}

# 6 unordered pairs (with repetition): 3 same-agent + 3 mixed. Index fixes
# the pair's seed_base. First element = seat A (Player1), second = seat B.
PAIRS: tuple = (
    ("random", "random"),
    ("random", "greedy_highest"),
    ("random", "shield_greedy"),
    ("greedy_highest", "greedy_highest"),
    ("greedy_highest", "shield_greedy"),
    ("shield_greedy", "shield_greedy"),
)

DEFAULT_OUTPUT = Path("ml/runs/baselines/baselines.json")


def seed_base_for(pair_index: int, campaign_length: int) -> int:
    """Distinct deterministic seed base per pair x campaign length."""
    return pair_index * 10_000 + campaign_length * 1_000


def play_episode(
    env: RegicideGymEnv, seat_agents: tuple, seed: int
) -> bool:
    """One episode: seat A/B act via their scripted agent. True iff team wins.

    Seeds BOTH RNG stacks with ``seed``: env.reset(seed=) re-seeds global
    ``random`` (engine shuffles); np.random is seeded here for RandomAgent.
    """
    np.random.seed(seed)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        active = env.engine.game_state.get_active_player()
        agent = seat_agents[0] if active.name == "Player1" else seat_agents[1]
        action = agent.act(env, obs, info)
        obs, _, terminated, truncated, info = env.step(int(action))
    return bool(terminated and info.get("result") == "win")


def evaluate_pair(
    seat_agents: tuple,
    campaign_length: int,
    n_episodes: int,
    seed_base: int,
) -> dict:
    """Win-rate record for one pair at one campaign length (+ Wilson LB)."""
    wins = 0
    for i in range(n_episodes):
        env = RegicideGymEnv(EnvConfig(campaign_length=campaign_length))
        try:
            if play_episode(env, seat_agents, seed_base + i):
                wins += 1
        finally:
            env.close()
    return {
        "win_rate": wins / n_episodes,
        "wins": wins,
        "episodes": n_episodes,
        "wilson_lb": wilson_lower_bound(wins, n_episodes),
        "seed_base": seed_base,
    }


def measure_all(n_episodes: int = EPISODES) -> dict:
    """Full measurement: the 6 pairs x campaign lengths, JSON-ready."""
    pairs: dict = {}
    for pair_index, (name_a, name_b) in enumerate(PAIRS):
        seat_agents: tuple = (AGENTS[name_a](), AGENTS[name_b]())
        pair_key = f"{name_a}+{name_b}"
        pairs[pair_key] = {
            str(length): evaluate_pair(
                seat_agents, length, n_episodes, seed_base_for(pair_index, length)
            )
            for length in CAMPAIGN_LENGTHS
        }
    return {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "episode_count": n_episodes,
            "campaign_lengths": list(CAMPAIGN_LENGTHS),
            "agent_classes": {name: cls.__name__ for name, cls in AGENTS.items()},
            "seat_order": "first name in key = seat A (Player1), second = seat B (Player2)",
            "seed_formula": "seed_base = pair_index * 10000 + campaign_length * 1000; "
            "episode seed = seed_base + i; pair_index is the 0-based index in pairs_order",
            "pairs_order": [f"{a}+{b}" for a, b in PAIRS],
        },
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure scripted-baseline team win rates (Step 2.3)"
    )
    parser.add_argument(
        "--episodes", type=int, default=EPISODES, help="Episodes per pair x length"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path"
    )
    args = parser.parse_args()

    report = measure_all(n_episodes=args.episodes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    print(f"wrote {args.output}")
    header = f"{'pair':40s} {'L=1 win_rate':>14s} {'L=3 win_rate':>14s}"
    print(header)
    print("-" * len(header))
    for pair_key, lengths in report["pairs"].items():
        print(
            f"{pair_key:40s} {lengths['1']['win_rate']:>10.3f} "
            f"(lb {lengths['1']['wilson_lb']:.3f})   "
            f"{lengths['3']['win_rate']:>6.3f} (lb {lengths['3']['wilson_lb']:.3f})"
        )


if __name__ == "__main__":
    main()
