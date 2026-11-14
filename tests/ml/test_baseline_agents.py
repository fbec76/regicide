"""Tests for Step 2.1 — scripted baselines + common trainer utilities.

Baseline agents are exercised on crafted game states: after a seeded reset we
overwrite the active player's hand (and the current enemy where needed) with
known cards, then ask the agent to act against the env's freshly rebuilt
legality mask (``env.current_mask()``). This keeps the assertions exact without
depending on random deals.
"""
import numpy as np
import pytest

from game.card import Card
from game.enemy import Enemy

from ml.trainers.baselines import (
    GreedyHighestAgent,
    RandomAgent,
    ShieldGreedyAgent,
)
from ml.trainers.common import evaluate_policy, make_env, wilson_lower_bound
from ml.envs.regicide_env import EnvConfig, RegicideGymEnv

CAMPAIGN_LENGTH = 1
N_EPISODES_PER_AGENT = 3


def _card(suit: str, rank: str, value: int) -> Card:
    return Card(suit, rank, value, "number", f"{value} of {suit.capitalize()}")


def _hearts_jack() -> Enemy:
    # Unshielded, non-spade enemy: spade suit power fires.
    return Enemy("hearts", "J", 20, "royal", "Jack of Hearts", health=20, attack=10)


def _fresh_attack_env(seed: int = 0) -> RegicideGymEnv:
    env = make_env(EnvConfig(campaign_length=CAMPAIGN_LENGTH), seed=seed, rank=0)
    env.reset(seed=seed)
    assert env.engine.phase == "attack"
    return env


def _set_hand(env: RegicideGymEnv, cards: list[Card]) -> dict:
    """Overwrite the active hand with known cards; return a fresh mask info."""
    env.engine.game_state.get_active_player().hand = list(cards)
    return {"mask": env.current_mask()}


AGENTS = [RandomAgent(), GreedyHighestAgent(), ShieldGreedyAgent()]


# ---------------------------------------------------------------------- #
# Legality by construction
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("agent", AGENTS, ids=[a.name for a in AGENTS])
def test_baselines_only_choose_unmasked_actions(agent) -> None:
    for episode in range(N_EPISODES_PER_AGENT):
        env = make_env(
            EnvConfig(campaign_length=CAMPAIGN_LENGTH), seed=episode, rank=episode
        )
        obs, info = env.reset(seed=episode)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            action = agent.act(env, obs, info)
            assert info["mask"][action], (
                f"{agent.name} chose masked action {action} "
                f"(episode {episode}, step {steps})"
            )
            obs, _, terminated, truncated, info = env.step(action)
            steps += 1
            assert steps <= 1000
        env.close()


# ---------------------------------------------------------------------- #
# GreedyHighest
# ---------------------------------------------------------------------- #
def test_greedy_highest_prefers_max_value_legal_play() -> None:
    agent = GreedyHighestAgent()

    # Single beats a low card: 10 of hearts (hand[0] -> single id 1) over 2 of spades.
    env = _fresh_attack_env(seed=1)
    info = _set_hand(env, [_card("hearts", "10", 10), _card("spades", "2", 2)])
    assert agent.act(env, None, info) == 1

    # A same-rank combo outscores any single: 3+3+3 = 9 beats value 3.
    env = _fresh_attack_env(seed=2)
    info = _set_hand(
        env,
        [
            _card("hearts", "3", 3),
            _card("spades", "3", 3),
            _card("diamonds", "3", 3),
        ],
    )
    action = agent.act(env, None, info)
    cards = env.engine.game_state.get_active_player().hand
    decoded = agent._decoded_cards(action, cards, "attack", None)
    assert sum(c.value for c in decoded) == 9
    env.close()


# ---------------------------------------------------------------------- #
# ShieldGreedy
# ---------------------------------------------------------------------- #
def test_shield_greedy_prefers_spades_when_unshielded() -> None:
    shield_agent = ShieldGreedyAgent()
    greedy_agent = GreedyHighestAgent()
    env = _fresh_attack_env(seed=3)
    env.engine.game_state.current_enemy = _hearts_jack()
    info = _set_hand(
        env,
        [
            _card("hearts", "10", 10),
            _card("spades", "2", 2),
            _card("spades", "3", 3),
        ],
    )
    # Enemy unshielded and non-spade: the spade-containing play wins even
    # though the 10 of hearts has higher raw value.
    assert shield_agent.act(env, None, info) == 3  # hand[2] -> 3 of spades
    # Plain GreedyHighest has no such preference.
    assert greedy_agent.act(env, None, info) == 1  # hand[0] -> 10 of hearts
    env.close()


# ---------------------------------------------------------------------- #
# Evaluation harness
# ---------------------------------------------------------------------- #
class _FirstLegalPolicy:
    """Deterministic scripted policy adapter (picks the lowest legal id)."""

    def predict(self, obs, action_masks=None, deterministic=True):
        legal = np.flatnonzero(np.asarray(action_masks, dtype=bool))
        return int(legal[0]), None


def test_evaluate_is_reproducible_with_seeded_env() -> None:
    first = evaluate_policy(
        _FirstLegalPolicy(), campaign_length=CAMPAIGN_LENGTH, n_episodes=5, seed_base=42
    )
    second = evaluate_policy(
        _FirstLegalPolicy(), campaign_length=CAMPAIGN_LENGTH, n_episodes=5, seed_base=42
    )
    assert first == second
    assert set(first) == {"win_rate", "wins", "episodes", "campaign_length"}
    assert first["episodes"] == 5
    assert 0.0 <= first["win_rate"] <= 1.0


def test_wilson_lower_bound_stays_below_point_estimate() -> None:
    assert wilson_lower_bound(0, 10) == 0.0
    # phat = 1 -> lower bound 1 / (1 + z^2/n), the standard all-success case.
    assert wilson_lower_bound(10, 10) == pytest.approx(1.0 / (1.0 + 1.96**2 / 10))
    assert 0.0 < wilson_lower_bound(5, 10) < 0.5
    with pytest.raises(ValueError):
        wilson_lower_bound(0, 0)
