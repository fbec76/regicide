import pytest
from unittest.mock import MagicMock, patch

from game.engine import Engine, handle_enemy_attack, create_enemy_faces, deal_initial_hands
from game.player import Player

from game.enemy import Enemy


class DummyLogger:
    def info(self, msg): pass

    def warning(self, msg): pass

    def error(self, msg): pass

    def data(self, msg): pass


@pytest.fixture
def dummy_logger():
    return DummyLogger()


@pytest.fixture
def engine(dummy_logger):
    # get_logger is not used in Engine, but required by signature
    return Engine(dummy_logger, lambda name=None: dummy_logger)


def test_create_enemy_faces(engine):
    jacks, queens, kings = create_enemy_faces()
    assert len(jacks) == 4
    assert len(queens) == 4
    assert len(kings) == 4
    assert all(isinstance(e, Enemy) for e in jacks + queens + kings)


def test_deal_initial_hands(engine):
    players = [Player("A", 2), Player("B", 2)]
    cards = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]
    tavern_deck = MagicMock()
    tavern_deck.draw_one.side_effect = cards
    gs = MagicMock()
    gs.players = players
    gs.tavern_deck = tavern_deck
    deal_initial_hands(gs)
    assert all(len(p.hand) == 2 for p in players)


def test_handle_enemy_attack_survives(monkeypatch, engine):
    player = MagicMock()
    player.hand_size.return_value = 2
    player.hand = [MagicMock(value=2), MagicMock(value=2)]
    player.name = "TestPlayer"
    player.play_cards.return_value = True
    gs = MagicMock()
    gs.current_enemy.effective_attack.return_value = 2
    gs.discard_pile = MagicMock()
    # Simulate user entering '0 1' for discards
    monkeypatch.setattr("builtins.input", lambda _: "0 1")
    with patch("game.engine.show_discard_phase") as show_discard_phase:
        assert handle_enemy_attack(gs, player)
        show_discard_phase.assert_called()


def test_handle_enemy_attack_defeated(engine):
    player = MagicMock()
    player.hand_size.return_value = 0
    player.name = "TestPlayer"
    gs = MagicMock()
    gs.current_enemy.effective_attack.return_value = 2
    # Simulate user input (not needed, but for completeness)
    with patch("game.engine.show_discard_phase"), patch("game.engine.show_loss") as show_loss:
        assert not handle_enemy_attack(gs, player)
        show_loss.assert_called()


def test_log_data_state(engine, dummy_logger):
    player = MagicMock()
    player.name = "A"
    player.hand = [MagicMock()]
    gs = MagicMock()
    gs.players = [player]
    gs.tavern_deck.cards = [MagicMock()]
    gs.castle_deck.cards = [MagicMock()]
    gs.discard_pile.cards = [MagicMock()]
    engine.log_data_state(gs)
    # No assertion, just ensure no error


def test_handle_play_action_enemy_defeated(monkeypatch, engine):
    # Setup mocks for player, game_state, and cards
    player = MagicMock()
    player.name = "TestPlayer"
    player.hand = [MagicMock(value=2, suit='hearts', c_type='number')]
    player.play_cards.return_value = True
    game_state = MagicMock()
    game_state.current_enemy.suit = 'spades'
    game_state.enemy_immunity = False
    game_state.face_jester_played = False
    game_state.played_this_enemy = []
    game_state.castle_deck.draw_one.return_value = None
    game_state.current_enemy.is_defeated.return_value = True
    game_state.current_enemy.current_damage = 20
    game_state.current_enemy.max_health = 20
    game_state.current_enemy.display_name = "Jack of Spades"
    game_state.get_active_player.return_value = player
    game_state.update_enemy_immunity.return_value = None
    game_state.players = [player]
    played_spades = []
    # Patch input and prompt_card_indexes_to_play
    monkeypatch.setattr("builtins.input", lambda _: "0")
    with patch("game.engine.prompt_card_indexes_to_play", return_value=player.hand), \
         patch("game.engine.is_valid_play", return_value=True), \
         patch("game.engine.show_invalid_action"), \
         patch("game.engine.can_play_jester", return_value=False), \
         patch("game.engine.show_played_cards"), \
         patch("game.engine.is_enemy_immune", return_value=False), \
         patch("game.engine.resolve_suit_power"), \
         patch("game.engine.total_attack", return_value=20), \
         patch("game.engine.exact_defeat", return_value=True), \
         patch("game.engine.show_loss"), \
         patch("game.engine.show_win"):
        result = engine.handle_play_action(player, game_state, played_spades)
        assert result in ('enemy_defeated', 'game_won')


def test_handle_yield_action_continue(engine):
    player = MagicMock()
    player.name = "TestPlayer"
    game_state = MagicMock()
    result = engine.handle_yield_action(player, game_state)
    assert result == 'continue'
