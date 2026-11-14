"""Shared fixtures for the E2E regression suite.

This suite is the regression contract for the later consolidation phase: every
test drives the real Engine/GameState/decks with only valid actions and checks
card conservation after every step. It must pass against the current code and
keep passing after the planned rule fixes.

All game randomness flows through the global ``random`` module (Deck.shuffle,
CastleDeck.create_castle_stack, TavernDeck.__init__ and DiscardPile.
shuffle_and_return_n_cards), so seeding it makes whole runs deterministic.
"""
import builtins
import random

import pytest

from game.engine import Engine
from game.player import Player


class DummyLogger:
    """No-op logger mirroring the DummyLogger pattern in tests/test_engine.py."""

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def data(self, *args, **kwargs):
        pass


@pytest.fixture(autouse=True)
def _silence_output(monkeypatch):
    """display.py, rules.py (suit powers) and engine.step_defense print
    directly to stdout; silence them so pytest output stays readable."""
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)


@pytest.fixture
def dummy_logger():
    return DummyLogger()


@pytest.fixture
def make_engine(dummy_logger):
    """Factory for a fully deterministic 2-player engine.

    2 players => tavern 40 + castle 12 = 52 cards and 0 Jesters, so the gym
    jester prompt_next_player input() blocker (engine.handle_jester_played)
    is never hit.
    """

    def _make_engine(seed):
        random.seed(seed)
        players = [Player(name="P1", hand_limit=8), Player(name="P2", hand_limit=8)]
        engine = Engine(dummy_logger, lambda name=None: dummy_logger)
        engine.reset(players)
        return engine

    return _make_engine


@pytest.fixture
def total_cards():
    """Card conservation invariant: hands + tavern + castle + discard + the
    enemy in play + cards played against the enemy must always equal 52."""

    def _total_cards(game_state):
        hands = sum(p.hand_size() for p in game_state.players)
        return (
            hands
            + len(game_state.tavern_deck.cards)
            + len(game_state.castle_deck.cards)
            + len(game_state.discard_pile.cards)
            + (1 if game_state.current_enemy else 0)
            + len(game_state.cards_against_enemy)
        )

    return _total_cards
