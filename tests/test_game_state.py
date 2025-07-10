from collections import deque
from game.game_state import GameState
from game.player import Player
from game.card import Card

class DummyDeck:
    def __init__(self, cards):
        self.cards = deque(cards)
    def draw_one(self):
        return self.cards.popleft() if self.cards else None

class DummyPlayer(Player):
    def __init__(self, name, alive=True):
        self.name = name
        self._alive = alive
    def is_alive(self):
        return self._alive

def test_initial_state():
    players = [DummyPlayer('A'), DummyPlayer('B')]
    tavern = DummyDeck([Card('hearts', '2', 2, 'number', '2')])
    castle = DummyDeck([Card('spades', 'K', 13, 'face', 'King')])
    discard = DummyDeck([])
    gs = GameState(players, tavern, castle, discard)
    assert gs.get_active_player().name == 'A'
    assert gs.current_enemy.rank == 'K'
    assert gs.enemy_immunity is True
    assert gs.is_game_won() is False

def test_advance_turn():
    players = [DummyPlayer('A'), DummyPlayer('B')]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    gs.advance_turn()
    assert gs.get_active_player().name == 'B'
    assert gs.total_turns == 1

def test_update_enemy_immunity():
    players = [DummyPlayer('A')]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    gs.face_jester_played = True
    gs.update_enemy_immunity()
    assert gs.enemy_immunity is False
    gs.face_jester_played = False
    gs.update_enemy_immunity()
    assert gs.enemy_immunity is True

def test_can_yield():
    players = [DummyPlayer('A'), DummyPlayer('B')]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    # First turn, always can yield
    assert gs.can_yield(players[0]) is True
    # Simulate all others have yielded
    gs.total_turns = 1
    gs.yields_in_a_row = {p.name: True for p in players}
    assert gs.can_yield(players[0]) is False
    # At least one other has not yielded
    gs.yields_in_a_row[players[1].name] = False
    assert gs.can_yield(players[0]) is True

def test_check_loss_all_dead():
    players = [DummyPlayer('A', alive=False), DummyPlayer('B', alive=False)]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    assert gs.check_loss() is True

def test_check_loss_game_won():
    players = [DummyPlayer('A')]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    gs.current_enemy = None
    assert gs.check_loss() is False

def test_is_game_won():
    players = [DummyPlayer('A')]
    gs = GameState(players, DummyDeck([]), DummyDeck([Card('spades', 'K', 13, 'face', 'King')]), DummyDeck([]))
    assert gs.is_game_won() is False
    gs.current_enemy = None
    assert gs.is_game_won() is True
