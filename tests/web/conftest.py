"""Fixtures for bridge handler-level tests (no wire, real Engine)."""
import builtins
import os
import random
import sys

import pytest

# sys.path bootstrap mirroring tests/conftest.py (one level deeper).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from game.game_state import GameState  # noqa: E402
from web.websocket_cli_bridge import WebSocketCLIBridge  # noqa: E402

# 2 players: tavern 40 + castle 12 = 52 cards (0 Jesters).
FULL_CAMPAIGN_CARDS = 52


class DummyLogger:
    """No-op logger mirroring the DummyLogger pattern in tests/e2e/conftest.py."""

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
    directly to stdout. The bridge intentionally does not silence them
    (documented engine noise); tests mute stdout for readability."""
    monkeypatch.setattr(builtins, "print", lambda *args, **kwargs: None)


@pytest.fixture
def dummy_logger():
    return DummyLogger()


@pytest.fixture
def total_cards():
    """Card conservation invariant across ALL SIX buckets: hands + tavern +
    castle + discard + the enemy in play + cards against the enemy == 52."""

    def _total_cards(game_state: GameState) -> int:
        return (
            sum(p.hand_size() for p in game_state.players)
            + len(game_state.tavern_deck.cards)
            + len(game_state.castle_deck.cards)
            + len(game_state.discard_pile.cards)
            + (1 if game_state.current_enemy else 0)
            + len(game_state.cards_against_enemy)
        )

    return _total_cards


class RecordingTransport:
    """Fake transport callbacks recording per-client sends and broadcasts."""

    def __init__(self):
        self.sent = {}  # client_id -> list[message]
        self.broadcasts = []  # list[message]

    async def send_to_client(self, client_id, message):
        self.sent.setdefault(client_id, []).append(message)

    async def broadcast(self, session_id, message):
        self.broadcasts.append(message)

    def to_client(self, client_id):
        return self.sent.get(client_id, [])

    def clear(self):
        self.sent.clear()
        self.broadcasts.clear()


@pytest.fixture
def transport():
    return RecordingTransport()


@pytest.fixture
def make_bridge(dummy_logger, transport):
    """Factory: bridge with fake broadcast/send callbacks and a real Engine
    created inside initialize_game(). Uses the shared ``transport`` fixture
    so tests can assert on recorded messages. Seeding drives all randomness."""

    def _make_bridge(client_players, seed=None):
        if seed is not None:
            random.seed(seed)
        return WebSocketCLIBridge(
            "test-session",
            client_players,
            transport.broadcast,
            transport.send_to_client,
            dummy_logger,
        )

    return _make_bridge
