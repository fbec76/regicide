"""Tests for ml/envs/reward.py — PBRS potential and reward computation.

Fixtures mirror tests/test_engine.py patterns: real Engine.reset states
manipulated via deepcopy + direct attribute mutation, and hand-built
GameState objects with real Player/Enemy/Deck objects.
"""

import ast
import inspect
from copy import deepcopy
from pathlib import Path

import pytest

from game.card import Card
from game.deck import Deck
from game.engine import Engine
from game.enemy import Enemy
from game.game_state import GameState
from game.player import Player

from ml.envs import reward as reward_module
from ml.envs.reward import (
    AGGRESSIVE_SHAPING,
    MINIMAL_SHAPING,
    SPARSE_CURRICULUM,
    compute_reward,
    potential,
)

GAMMA = 0.99  # MUST match the training config discount (PBRS requirement)


class DummyLogger:
    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass

    def data(self, msg):
        pass


def make_engine():
    return Engine(DummyLogger(), lambda name=None: DummyLogger())


def make_card(value, suit="hearts"):
    return Card(suit, str(value), value, "number", f"{value} of {suit.capitalize()}")


def make_enemy(health=20, damage=0, attack=10, suit="hearts", rank="J"):
    enemy = Enemy(suit, rank, 10, "enemy", f"{rank} of {suit.capitalize()}", health, attack)
    enemy.current_damage = damage
    return enemy


def make_gs(hands, enemy=None, castle_remaining=11, tavern_count=40, discard_count=0):
    """Hand-built GameState with real Player/Enemy/Deck objects.

    castle_remaining is the castle size AFTER construction: an enemy in play
    requires castle_remaining + 1 cards at construction (GameState draws one);
    enemy=None builds a won state (empty castle -> current_enemy is None).
    """
    players = [Player(f"P{i}", 7) for i in range(len(hands))]
    for player, values in zip(players, hands):
        player.hand = [make_card(v) for v in values]
    castle_count = 0 if enemy is None else castle_remaining + 1
    castle = Deck([Card("spades", "Q", 15, "enemy", f"Filler {i}") for i in range(castle_count)])
    tavern = Deck([make_card(2, "clubs") for _ in range(tavern_count)])
    discard = Deck([make_card(3, "diamonds") for _ in range(discard_count)])
    gs = GameState(players=players, tavern_deck=tavern, castle_deck=castle, discard_pile=discard)
    if enemy is not None:
        gs.current_enemy = enemy
    return gs


def test_pbrs_potential_loop_is_zero_sum():
    """The return-discounted shaped sum Σ γ^i·(γΦ(s_{i+1})−Φ(s_i)) over
    non-terminal transitions telescopes exactly to γ^T·Φ(s_T) − Φ(s_0) —
    no extra terms may leak into the shaped delta. (The identity holds for
    the discounted sum, as shaped rewards enter the return with factor γ^i;
    an undiscounted sum would carry a (γ−1)·Σ_mid Φ residue.)"""
    engine = make_engine()
    engine.reset([Player("A", 8), Player("B", 8)])
    states = [deepcopy(engine.game_state)]
    for damage, discards in ((5, 1), (10, 2), (12, 3)):
        nxt = deepcopy(states[-1])
        nxt.current_enemy.current_damage = damage
        for _ in range(discards):
            nxt.discard_pile.add_to_bottom(make_card(4, "clubs"))
        states.append(nxt)
    assert all(not s.is_game_won() for s in states)

    phis = [potential(s, AGGRESSIVE_SHAPING) for s in states]
    snapshots = deepcopy(states)
    total = 0.0
    for i in range(len(states) - 1):
        step_reward = compute_reward(
            states[i], states[i + 1], phis[i], AGGRESSIVE_SHAPING, GAMMA, {}
        )
        total += (GAMMA**i) * step_reward
    steps = len(states) - 1
    assert total == pytest.approx(GAMMA**steps * phis[-1] - phis[0])

    # Purity: compute_reward mutated nothing (enemy damage, piles unchanged).
    for before, after in zip(snapshots, states):
        assert after.current_enemy.current_damage == before.current_enemy.current_damage
        assert len(after.discard_pile.cards) == len(before.discard_pile.cards)
        assert len(after.tavern_deck.cards) == len(before.tavern_deck.cards)
        assert [c.value for p in after.players for c in p.hand] == [
            c.value for p in before.players for c in p.hand
        ]


def test_terminal_phi_is_zero():
    """Φ(terminal) = 0: won states return 0 from potential(); loss states
    (no GameState marker) are forced to 0 via events['terminal'], and the
    terminal ±10 is added ON TOP of the collapsed shaped delta −Φ(s_prev)."""
    won = make_gs(hands=([5], [3]), enemy=None, castle_remaining=0)
    assert won.is_game_won() is True
    for config in (AGGRESSIVE_SHAPING, MINIMAL_SHAPING, SPARSE_CURRICULUM):
        assert potential(won, config) == 0.0

    prev = make_gs(hands=([5], [3]), enemy=make_enemy(damage=10), castle_remaining=3)
    phi_prev = potential(prev, AGGRESSIVE_SHAPING)

    # Win terminal: shaped part collapses to −Φ(s_prev), +10 on top.
    win = compute_reward(prev, won, phi_prev, AGGRESSIVE_SHAPING, GAMMA, {"terminal": "win"})
    assert win == pytest.approx(10.0 - phi_prev)
    # Win detected from the state alone (no event): Φ(s′)=0 still holds, no ±10.
    assert compute_reward(prev, won, phi_prev, AGGRESSIVE_SHAPING, GAMMA, {}) == pytest.approx(-phi_prev)

    # Loss terminal: next state looks mid-game, the event forces Φ(s′)=0.
    loss_next = make_gs(hands=([5], [3]), enemy=make_enemy(damage=0), castle_remaining=3)
    loss = compute_reward(prev, loss_next, phi_prev, AGGRESSIVE_SHAPING, GAMMA, {"terminal": "loss"})
    assert loss == pytest.approx(-10.0 - phi_prev)


def test_exact_defeat_bonus_outside_phi():
    """The +1 exact-defeat bonus is added outside Φ: it never enters the
    potential function and appears only when events say so."""
    config = AGGRESSIVE_SHAPING
    prev = make_gs(hands=([5], [3]), enemy=make_enemy(health=20, damage=15), castle_remaining=3)
    nxt = make_gs(hands=([2], [3]), enemy=make_enemy(health=20, damage=0), castle_remaining=2)
    phi_prev = potential(prev, config)

    base = compute_reward(prev, nxt, phi_prev, config, GAMMA, {})
    assert base == pytest.approx(GAMMA * potential(nxt, config) - phi_prev)

    with_bonus = compute_reward(prev, nxt, phi_prev, config, GAMMA, {"exact_defeat": True})
    assert with_bonus - base == pytest.approx(config.exact_defeat_bonus)
    assert with_bonus - base == pytest.approx(1.0)

    # With sparse shaping (Φ ≡ 0) the bonus is the entire reward.
    sparse = compute_reward(prev, nxt, 0.0, SPARSE_CURRICULUM, GAMMA, {"exact_defeat": True})
    assert sparse == pytest.approx(1.0)


def test_reward_configs_match_spec():
    assert (AGGRESSIVE_SHAPING.name, AGGRESSIVE_SHAPING.w1, AGGRESSIVE_SHAPING.w2,
            AGGRESSIVE_SHAPING.w3, AGGRESSIVE_SHAPING.w4, AGGRESSIVE_SHAPING.w5) == \
        ("aggressive_shaping", 2.0, 1.0, 0.3, 0.2, 0.1)
    assert (MINIMAL_SHAPING.name, MINIMAL_SHAPING.w1, MINIMAL_SHAPING.w2,
            MINIMAL_SHAPING.w3, MINIMAL_SHAPING.w4, MINIMAL_SHAPING.w5) == \
        ("minimal_shaping", 1.0, 0.5, 0.0, 0.0, 0.0)
    assert (SPARSE_CURRICULUM.name, SPARSE_CURRICULUM.w1, SPARSE_CURRICULUM.w2,
            SPARSE_CURRICULUM.w3, SPARSE_CURRICULUM.w4, SPARSE_CURRICULUM.w5) == \
        ("sparse_curriculum", 0.0, 0.0, 0.0, 0.0, 0.0)
    for config in (AGGRESSIVE_SHAPING, MINIMAL_SHAPING, SPARSE_CURRICULUM):
        assert config.exact_defeat_bonus == 1.0
        assert config.win_reward == 10.0
        assert config.loss_reward == -10.0


def test_engine_reward_never_used():
    """Structural: reward.py never imports game.engine (engine rewards never
    pass through) and its public API takes GameState snapshots + floats only."""
    source = Path(reward_module.__file__).read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(not alias.name.startswith("game.engine") for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            assert not node.module.startswith("game.engine")

    assert list(inspect.signature(potential).parameters) == ["gs", "config"]
    assert list(inspect.signature(compute_reward).parameters) == [
        "prev_gs", "next_gs", "prev_potential", "config", "gamma", "events",
    ]


def test_gamma_out_of_range_rejected():
    """gamma must equal the algorithm discount — values outside [0, 1] fail loudly."""
    prev = make_gs(hands=([5], [3]), enemy=make_enemy(), castle_remaining=3)
    nxt = make_gs(hands=([5], [3]), enemy=make_enemy(damage=5), castle_remaining=3)
    with pytest.raises(ValueError):
        compute_reward(prev, nxt, 0.0, AGGRESSIVE_SHAPING, 1.5, {})
    with pytest.raises(ValueError):
        compute_reward(prev, nxt, 0.0, AGGRESSIVE_SHAPING, -0.1, {})
