"""Tests for ml/envs/regicide_env.py — Step 1.5 Gym env contract.

Named tests from the plan: gym API 5-tuple, env_checker, episode termination,
step-limit truncation, campaign truncation, stdout silence, injected jester
chooser, mask-in-info, plus the R1 (defense auto-loss) and R2 (stuck mask)
review amendments.
"""

import builtins

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from game.card import Card

from ml.envs.action_mask import N_ACTIONS
from ml.envs.regicide_env import EnvConfig, RegicideGymEnv


def make_zero_card():
    """A value-0 hearts 'number' card: no damage, no heal, no suit effect."""
    return Card("hearts", "0", 0, "number", "Zero of Hearts")


def play_episode(env: RegicideGymEnv, seed: int, policy="random"):
    """Step a fresh episode with legal-mask-sampled actions until termination.

    Returns (steps, terminated, truncated, info) of the final step.
    """
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    steps = 0
    for _ in range(env.config.step_limit):
        mask = info["mask"]
        legal = np.flatnonzero(mask)
        if policy == "zero_single":
            # Play the lowest-value single (minimal damage/defensive burn).
            singles = legal[(legal >= 1) & (legal <= 7)]
            action = int(singles[0]) if len(singles) else int(legal[0])
        else:
            action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return steps, terminated, truncated, info


# --------------------------------------------------------------------- #
# Gym API contract
# --------------------------------------------------------------------- #
def test_gym_api_returns_5_tuple():
    env = RegicideGymEnv(EnvConfig(seed=0))
    obs, info = env.reset(seed=0)
    assert obs.shape == (108,) and obs.dtype == np.float32
    mask = info["mask"]
    obs, reward, terminated, truncated, info = env.step(int(np.flatnonzero(mask)[0]))
    assert len((obs, reward, terminated, truncated, info)) == 5
    assert isinstance(terminated, bool) and isinstance(truncated, bool)
    assert isinstance(reward, float)


def test_env_checker_passes():
    env = RegicideGymEnv(EnvConfig(seed=42))
    check_env(env, skip_render_check=True)


# --------------------------------------------------------------------- #
# Episode dynamics
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_episode_terminates_win_or_loss(seed):
    env = RegicideGymEnv(EnvConfig(seed=seed))
    steps, terminated, truncated, info = play_episode(env, seed=seed)
    assert terminated or truncated
    assert steps <= env.config.step_limit
    if terminated:
        assert info["result"] in ("win", "player_defeated")


def test_step_limit_truncates_at_1000():
    # step_limit=20 override per plan; campaign_length=12 makes a win
    # impossible in 20 steps, and the zero-attack/zero-card setup below
    # makes a loss impossible — the episode can only time-limit truncate.
    config = EnvConfig(campaign_length=12, step_limit=20, seed=11)
    env = RegicideGymEnv(config)
    obs, info = env.reset(seed=11)
    game_state = env.engine.game_state
    # Zero-attack enemy + a refilled value-0 hand: every step is a
    # damage-less play that advances the turn — nothing can terminate.
    game_state.current_enemy.attack = 0
    zero = make_zero_card()
    for _ in range(config.step_limit):
        game_state.get_active_player().hand = [zero]
        obs, reward, terminated, truncated, info = env.step(1)  # single: slot 0
    assert truncated is True and terminated is False
    assert info["result"] == "turn_advanced"


def test_campaign_truncation_yields_L_enemies_total():
    for length in (1, 3):
        env = RegicideGymEnv(EnvConfig(campaign_length=length, seed=8))
        env.reset(seed=8)
        gs = env.engine.game_state
        in_play = 1 if gs.current_enemy is not None else 0
        total = len(gs.castle_deck.cards) + in_play
        assert total == length


def test_stdout_silent_during_step_reset(capsys):
    env = RegicideGymEnv(EnvConfig(seed=3))
    env.reset(seed=3)
    assert capsys.readouterr().out == ""
    rng = np.random.default_rng(0)
    for _ in range(10):
        legal = np.flatnonzero(env.action_masks())
        _, _, terminated, truncated, info = env.step(int(rng.choice(legal)))
        if terminated or truncated:
            break
    assert capsys.readouterr().out == ""


def test_jester_uses_injected_chooser_no_input(monkeypatch):
    env = RegicideGymEnv(EnvConfig(seed=5))
    obs, info = env.reset(seed=5)
    gs = env.engine.game_state
    # Force a jester into the active hand (2-player decks hold none).
    gs.get_active_player().hand[0] = Card("joker", "Jester", 0, "jester", "Jester")

    def forbidden(*args, **kwargs):
        raise AssertionError("input() must not be called — chooser must be injected")

    monkeypatch.setattr(builtins, "input", forbidden)
    obs, reward, terminated, truncated, info = env.step(1)  # single: hand slot 0
    assert info["result"] == "jester"
    assert not terminated


# --------------------------------------------------------------------- #
# Mask plumbing
# --------------------------------------------------------------------- #
def test_mask_present_in_info():
    env = RegicideGymEnv(EnvConfig(seed=6))
    obs, info = env.reset(seed=6)
    mask = info["mask"]
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (N_ACTIONS,) and mask.dtype == np.bool_
    legal = np.flatnonzero(mask)
    obs, reward, terminated, truncated, info = env.step(int(legal[0]))
    assert "mask" in info and "action_mask" in info
    assert np.array_equal(info["mask"], info["action_mask"])
    assert info["campaign_length"] == 3
    if not terminated:
        assert info["mask"].sum() > 0


# --------------------------------------------------------------------- #
# R1 — defense auto-loss on insufficient hand
# --------------------------------------------------------------------- #
def test_defense_insufficient_hand_is_terminal():
    env = RegicideGymEnv(EnvConfig(seed=13))
    env.reset(seed=13)
    gs = env.engine.game_state
    # Construct an unplayable defense state: a weak hand against an enemy
    # attack (J/Q/K attack is 10/15/20 — all > 2).
    gs.get_active_player().hand.clear()
    gs.get_active_player().add_card(Card("hearts", "2", 2, "number", "2 of Hearts"))
    env.engine.phase = "defense"
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True and truncated is False
    assert info["result"] == "player_defeated"
    assert reward < 0


# --------------------------------------------------------------------- #
# R2 — stuck state: mask is YIELD-only and resolves the loss
# --------------------------------------------------------------------- #
def test_stuck_state_mask_not_empty():
    env = RegicideGymEnv(EnvConfig(seed=21))
    env.reset(seed=21)
    gs = env.engine.game_state
    active = gs.get_active_player()
    active.hand.clear()
    for player in gs.players:
        if player is not active:
            gs.yields_in_a_row[player.name] = True  # yield forbidden for active
    mask = env.action_masks()
    assert mask.sum() == 1 and mask[0]  # exactly YIELD unmasks (R2 carve-out)
    obs, reward, terminated, truncated, info = env.step(0)
    assert terminated is True
    assert info["result"] == "player_defeated"
