import pytest
from unittest.mock import MagicMock, patch

from game.engine import Engine, create_enemy_faces, deal_initial_hands
from game.card import Card
from game.deck import DiscardPile
from game.player import Player

from game.enemy import Enemy


def make_card(suit, rank, value, c_type='number', display_name=None):
    return Card(suit, rank, value, c_type, display_name if display_name is not None else f"{rank} of {suit}")


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
        result = engine.handle_play_action(player, game_state)
        assert result in ('enemy_defeated', 'game_won')


def test_handle_yield_action_continue(engine):
    player = MagicMock()
    player.name = "TestPlayer"
    game_state = MagicMock()
    result = engine.handle_yield_action(player, game_state)
    assert result == 'continue'


def test_step_attack_rejects_cards_not_in_hand(engine):
    player = Player("A", 8)
    in_hand = make_card('clubs', '5', 5)
    player.add_card(in_hand)
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    game_state = MagicMock()
    game_state.get_active_player.return_value = player
    game_state.current_enemy = enemy
    engine.game_state = game_state
    engine.phase = "attack"
    foreign = make_card('clubs', '7', 7)
    obs, reward, done, info = engine.step_attack({"type": "play", "cards": [foreign]})
    assert reward == -1
    assert info == {"error": "Invalid play"}
    assert done is False
    assert player.hand == [in_hand]
    assert enemy.current_damage == 0


def test_step_attack_accepts_cards_in_hand_and_removes_them(engine):
    player = Player("A", 8)
    card = make_card('hearts', '5', 5)
    player.add_card(card)
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    game_state = MagicMock()
    game_state.get_active_player.return_value = player
    game_state.current_enemy = enemy
    game_state.cards_against_enemy = []
    engine.game_state = game_state
    engine.phase = "attack"
    with patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"):
        obs, reward, done, info = engine.step_attack({"type": "play", "cards": [card]})
    assert player.hand == []
    assert game_state.cards_against_enemy == [card]
    assert enemy.current_damage == 5
    assert reward == 0.2
    assert info == {"result": "defense_required"}
    assert done is False


def test_step_defense_rejects_discards_not_in_hand(engine):
    player = Player("A", 8)
    c1 = make_card('clubs', '8', 8)
    c2 = make_card('diamonds', '4', 4)
    player.add_card(c1)
    player.add_card(c2)
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    game_state = MagicMock()
    game_state.get_active_player.return_value = player
    game_state.current_enemy = enemy
    game_state.discard_pile = DiscardPile()
    engine.game_state = game_state
    engine.phase = "defense"
    foreign = make_card('clubs', '2', 2)
    obs, reward, done, info = engine.step_defense({"type": "defend", "cards": [foreign]})
    assert reward == -1
    assert info == {"error": "Invalid defense"}
    assert done is False
    assert player.hand == [c1, c2]
    assert game_state.discard_pile.cards == []


def test_step_defense_rejects_insufficient_total(engine):
    player = Player("A", 8)
    c1 = make_card('clubs', '8', 8)
    c2 = make_card('diamonds', '4', 4)
    player.add_card(c1)
    player.add_card(c2)
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    game_state = MagicMock()
    game_state.get_active_player.return_value = player
    game_state.current_enemy = enemy
    game_state.discard_pile = DiscardPile()
    engine.game_state = game_state
    engine.phase = "defense"
    obs, reward, done, info = engine.step_defense({"type": "defend", "cards": [c1]})
    assert reward == -1
    assert info == {"error": "Invalid defense"}
    assert done is False
    assert player.hand == [c1, c2]
    assert game_state.discard_pile.cards == []


def test_step_defense_uncoverable_still_loss(engine):
    player = Player("A", 8)
    player.add_card(make_card('clubs', '2', 2))
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 10)
    game_state = MagicMock()
    game_state.get_active_player.return_value = player
    game_state.current_enemy = enemy
    game_state.discard_pile = DiscardPile()
    engine.game_state = game_state
    engine.phase = "defense"
    with patch("game.engine.show_loss"):
        obs, reward, done, info = engine.step_defense({"type": "defend", "cards": []})
    assert reward == -10
    assert info == {"result": "player_defeated"}
    assert done is True
    assert player.hand != []
    assert game_state.discard_pile.cards == []


def test_step_attack_enemy_defeated_turn_behavior(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    active = game_state.get_active_player()
    # Drive the current Jack to exactly one 5-value card away from defeat.
    game_state.current_enemy.current_damage = game_state.current_enemy.max_health - 5
    killing_card = make_card('hearts', '5', 5)
    active.hand.append(killing_card)  # hand is at its limit after dealing; insert directly
    # The reset dealt the FIRST Jack as the current enemy, so three more Jacks
    # still precede the Queens. Move those remaining Jacks behind the
    # Queens/Kings so the next castle card is a Queen and the J->Q->K rank
    # order the assertion below relies on holds.
    remaining_cards = game_state.castle_deck.cards
    jacks_left = [c for c in remaining_cards if c.rank == 'J']
    game_state.castle_deck.cards = [c for c in remaining_cards if c.rank != 'J'] + jacks_left
    next_enemy = game_state.castle_deck.peek_top()
    assert next_enemy.rank == 'Q'  # castle stack order is Jacks -> Queens -> Kings
    with patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"), patch("game.engine.show_new_enemy"):
        obs, reward, done, info = engine.step_attack({"type": "play", "cards": [killing_card]})
    assert reward == 2
    assert info == {"result": "enemy_defeated"}
    assert done is False
    # Option A: the defeating player begins a new turn against the revealed
    # enemy — the turn does NOT advance to the other player.
    assert game_state.get_active_player() is active
    assert engine.phase == "attack"
    assert game_state.current_enemy is next_enemy
    assert game_state.current_enemy.current_damage == 0


def test_step_attack_win_terminal(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    active = game_state.get_active_player()
    # Make the LAST castle card (a King — stack order Jacks -> Queens -> Kings)
    # the current enemy and drop the remaining castle cards: defeating this
    # King empties the castle and wins the game.
    king = game_state.castle_deck.cards.pop()
    assert king.rank == 'K'
    game_state.castle_deck.cards = []
    game_state.current_enemy = king
    king.current_damage = king.max_health - 5
    killing_card = make_card('hearts', '5', 5)
    active.hand.append(killing_card)
    with patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"):
        obs, reward, done, info = engine.step_attack({"type": "play", "cards": [killing_card]})
    assert reward == 10
    assert done is True
    assert info == {"result": "win"}
    assert game_state.castle_deck.is_empty()
    assert game_state.is_game_won() is True


def test_run_enemy_defeated_turn_behavior(engine):
    # Deterministic CLI-path construction: create_enemy_faces is patched to
    # return three 5-health enemies, so each scripted 5-value play defeats the
    # current enemy outright and no defense phase is ever reached;
    # deal_initial_hands is patched out so hands are exactly the pre-set
    # cards. Player 0 holds exactly the two scripted cards (hand_limit 2):
    # they defeat enemies 1 and 2 on consecutive turns (Option A — the turn
    # does NOT advance), then auto-yield on turn 3 with their empty hand
    # (yield allowed, enemy attack 0), after which player 1 — empty-handed
    # with player 0 having yielded last turn — is STUCK on turn 4 and run()
    # terminates as a loss.
    players = [Player("A", 2), Player("B", 8)]
    enemies = [
        Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 5, 0),
        Enemy('hearts', 'Q', 15, 'enemy', 'Queen of Hearts', 5, 0),
        Enemy('spades', 'K', 20, 'enemy', 'King of Spades', 5, 0),
    ]
    scripted_cards = [make_card('hearts', '5', 5), make_card('hearts', '5', 5)]
    players[0].hand.extend(scripted_cards)
    prompt = MagicMock(side_effect=[[scripted_cards[0]], [scripted_cards[1]]])
    with patch("game.engine.create_enemy_faces", return_value=([enemies[0]], [enemies[1]], [enemies[2]])), \
            patch("game.engine.deal_initial_hands"), \
            patch("game.engine.prompt_card_indexes_to_play", prompt), \
            patch("game.engine.show_current_state"), patch("game.engine.show_player_hand"), \
            patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"), patch("game.engine.show_new_enemy"), \
            patch("game.engine.show_loss"):
        engine.run(players)
    game_state = engine.game_state
    # Turn never advanced to player 1 during the defeats: player 0 defeated
    # both enemies on consecutive turns (prompts 1 and 2), and the empty-handed
    # turns 3/4 never prompted.
    assert prompt.call_count == 2
    assert all(call.args[0] is players[0] for call in prompt.call_args_list)
    # The game ended on player 1's stuck turn 4.
    assert game_state.get_active_player() is players[1]
    assert game_state.current_enemy is enemies[2]  # castle advanced past both defeats
    assert game_state.is_game_won() is False


def test_step_attack_yield_records_flag(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    # Gym yield via an empty "play" action: the yielding player's flag is
    # recorded, the other player's stays False.
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert game_state.yields_in_a_row["A"] is True
    assert game_state.yields_in_a_row["B"] is False
    assert engine.phase == "defense"  # the Jack's attack (10) still needs covering


def test_step_attack_forbidden_yield_invalid(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    # Shield the enemy fully so a yield (and later B's play) ends the turn
    # without entering the defense phase.
    game_state.current_enemy.shield = game_state.current_enemy.attack
    # Turn 1: A yields (first turn of the game -> always allowed); turn
    # advances to B.
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert game_state.yields_in_a_row["A"] is True
    assert game_state.get_active_player() is players[1]
    # B attempts to yield while A's most recent turn was a yield -> forbidden,
    # no state mutation (turn not advanced, phase unchanged).
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert info == {"error": "Yield forbidden"}
    assert reward == pytest.approx(-1.01, abs=1e-9)  # -1 gate + -0.01 step penalty
    assert done is False
    assert game_state.get_active_player() is players[1]
    assert engine.phase == "attack"
    assert game_state.yields_in_a_row["B"] is False
    assert game_state.total_turns == 1
    # B plays a valid card instead (clubs: no suit power) -> turn advances to A.
    b_card = make_card('clubs', '2', 2)
    players[1].hand.append(b_card)
    obs, reward, done, info = engine.step({"type": "play", "cards": [b_card]})
    assert game_state.get_active_player() is players[0]
    # A yields: B's most recent turn was a play, so this is allowed again.
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert info == {"result": "yielded_turn_advanced"}
    assert game_state.yields_in_a_row["A"] is True


def test_step_stuck_player_loss(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    # Shield the enemy so A's yield advances the turn instead of requiring
    # a defense phase.
    game_state.current_enemy.shield = game_state.current_enemy.attack
    obs, reward, done, info = engine.step({"type": "play", "cards": []})  # A yields
    assert game_state.get_active_player() is players[1]
    # B's turn: drain B's hand — B is stuck (empty hand, and A yielded on
    # their most recent turn so B cannot yield either).
    players[1].hand = []
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    # Observable reward is -10.01: the stuck loss returns -10 from the
    # step() attack branch and the standard -0.01 dispatch penalty applies.
    assert reward == pytest.approx(-10.01, abs=1e-9)
    assert done is True
    assert info == {"result": "player_defeated"}
    assert engine.done is True
    assert game_state.get_active_player() is players[1]  # no turn advance


def test_step_empty_hand_survives_when_yield_allowed(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    game_state.current_enemy.shield = game_state.current_enemy.attack  # attack 0
    # A plays (does NOT yield), so A's most recent turn is a play.
    a_card = make_card('clubs', '2', 2)
    players[0].hand.append(a_card)
    obs, reward, done, info = engine.step({"type": "play", "cards": [a_card]})
    assert game_state.get_active_player() is players[1]
    # B's turn with an empty hand: yield allowed -> auto-yield, turn advances.
    players[1].hand = []
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert info == {"result": "yielded_turn_advanced"}
    assert reward == pytest.approx(-0.11, abs=1e-9)  # same reward as a normal yield
    assert done is False
    assert game_state.yields_in_a_row["B"] is True
    assert game_state.yields_in_a_row["A"] is False
    assert game_state.get_active_player() is players[0]
    assert engine.phase == "attack"


def test_step_empty_hand_shielded_enemy_survives(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    # Edge: empty hand + shield >= attack (effective attack 0) -> the
    # empty-handed player auto-yields and the turn advances; no defense phase.
    players[0].hand = []
    game_state.current_enemy.shield = game_state.current_enemy.attack
    obs, reward, done, info = engine.step({"type": "play", "cards": []})
    assert info == {"result": "yielded_turn_advanced"}
    assert reward == pytest.approx(-0.11, abs=1e-9)
    assert done is False
    assert engine.phase == "attack"  # no defense phase
    assert game_state.get_active_player() is players[1]
    assert game_state.yields_in_a_row["A"] is True


def test_run_empty_hand_auto_yields(engine):
    # Deterministic CLI construction: deal_initial_hands is patched out so
    # hands are exactly the pre-set cards; create_enemy_faces returns one
    # attack-0 enemy (health 40 survives all scripted damage), so no defense
    # phase is ever reached. A plays their only card (turn 1), B plays theirs
    # (turn 2); on turn 3 A's hand is empty but yield is allowed (B played
    # last turn) -> run() auto-yields WITHOUT prompting (prompt count stays
    # 2) and the turn advances; on turn 4 B is stuck (empty hand, A yielded
    # last turn) -> run() terminates as a loss.
    players = [Player("A", 8), Player("B", 8)]
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 40, 0)
    a_card = make_card('hearts', '5', 5)
    b_card = make_card('clubs', '2', 2)
    players[0].hand.append(a_card)
    players[1].hand.append(b_card)
    prompt = MagicMock(side_effect=lambda active_player: [active_player.hand[0]])
    with patch("game.engine.create_enemy_faces", return_value=([enemy], [], [])), \
            patch("game.engine.deal_initial_hands"), \
            patch("game.engine.prompt_card_indexes_to_play", prompt), \
            patch("game.engine.show_current_state"), patch("game.engine.show_player_hand"), \
            patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_loss"):
        engine.run(players)
    game_state = engine.game_state
    # The empty-handed turns (3 and 4) never prompted — auto-yield handled them.
    assert prompt.call_count == 2
    assert [call.args[0] for call in prompt.call_args_list] == [players[0], players[1]]
    # The auto-yield was recorded for A; the game ended on B's stuck turn.
    assert game_state.yields_in_a_row["A"] is True
    assert game_state.get_active_player() is players[1]
    assert game_state.current_enemy is enemy  # never defeated
    assert game_state.is_game_won() is False


def test_spade_vs_spades_enemy_with_immunity_pends(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    game_state.current_enemy = Enemy('spades', 'J', 10, 'enemy', 'Jack of Spades', 20, 10)
    spade5 = make_card('spades', '5', 5)
    players[0].hand.append(spade5)
    obs, reward, done, info = engine.step({"type": "play", "cards": [spade5]})
    # Immunity blocks the spade power: no shield, the card pends for the Jester.
    assert game_state.current_enemy.shield == 0
    assert game_state.pending_spades == [spade5]
    assert engine.phase == "defense"  # the unshielded attack still needs covering


def test_jester_applies_retroactive_shields(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    game_state.current_enemy = Enemy('spades', 'J', 10, 'enemy', 'Jack of Spades', 20, 10)
    spade5 = make_card('spades', '5', 5)
    players[0].hand.append(spade5)
    engine.step({"type": "play", "cards": [spade5]})  # immunity blocks it -> pends
    engine.phase = "attack"
    jester = make_card('joker', 'Jester', 0, 'jester')
    players[0].hand.append(jester)
    with patch("game.engine.show_jester_played"), \
            patch("game.engine.prompt_next_player", return_value=players[1]):
        obs, reward, done, info = engine.step({"type": "play", "cards": [jester]})
    # The pending spade retroactively shields the enemy once immunity is off.
    assert game_state.current_enemy.shield == 5
    assert game_state.pending_spades == []
    assert game_state.face_jester_played is True
    assert game_state.enemy_immunity is False
    assert info == {"result": "jester"}
    assert game_state.get_active_player() is players[1]


def test_pending_spades_cleared_on_defeat(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    enemy = Enemy('spades', 'J', 10, 'enemy', 'Jack of Spades', 20, 10)
    game_state.current_enemy = enemy
    # A spade from an earlier turn was blocked by immunity and pended.
    game_state.pending_spades.append(make_card('spades', '5', 5))
    enemy.current_damage = enemy.max_health - 5
    killing = make_card('spades', '5', 5)
    players[0].hand.append(killing)
    obs, reward, done, info = engine.step({"type": "play", "cards": [killing]})
    assert reward == pytest.approx(1.99, abs=1e-9)  # 2 defeat reward - 0.01 penalty
    assert info == {"result": "enemy_defeated"}
    assert done is False
    # Battle-scoped state resets with the enemy: nothing leaks to the next one.
    assert game_state.pending_spades == []
    new_enemy = game_state.current_enemy
    assert new_enemy is not enemy
    assert new_enemy.shield == 0
    assert new_enemy.current_damage == 0
    assert game_state.face_jester_played is False


def test_spade_resolves_normally_after_jester(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    game_state.current_enemy = Enemy('spades', 'J', 10, 'enemy', 'Jack of Spades', 20, 10)
    jester = make_card('joker', 'Jester', 0, 'jester')
    players[0].hand.append(jester)
    with patch("game.engine.show_jester_played"), \
            patch("game.engine.prompt_next_player", return_value=players[1]):
        engine.step({"type": "play", "cards": [jester]})
    assert game_state.current_enemy.shield == 0  # nothing pending to apply
    # The next player plays a spade: immunity is off, so it shields directly.
    spade3 = make_card('spades', '3', 3)
    players[1].hand.append(spade3)
    obs, reward, done, info = engine.step({"type": "play", "cards": [spade3]})
    assert game_state.current_enemy.shield == 3
    assert game_state.pending_spades == []  # not pended — resolved immediately
    assert info == {"result": "defense_required"}  # 10 - 3 = 7 attack remains


def test_exact_defeat_enemy_to_tavern_top(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    enemy1 = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 20, 0)
    game_state.current_enemy = enemy1
    enemy1.current_damage = enemy1.max_health - 5
    spade5a = make_card('spades', '5', 5)
    players[0].hand.append(spade5a)
    engine.step({"type": "play", "cards": [spade5a]})
    # Exact defeat (20 damage == 20 health): the enemy card goes to the tavern TOP.
    assert game_state.tavern_deck.peek_top() is enemy1
    assert enemy1 not in game_state.discard_pile.cards
    # Non-exact defeat: the enemy card goes to the discard pile instead.
    enemy2 = Enemy('diamonds', 'Q', 15, 'enemy', 'Queen of Diamonds', 24, 0)
    enemy2.current_damage = 20  # +5 damage = 25 >= 24, but 25 != 24 -> not exact
    game_state.current_enemy = enemy2
    spade5b = make_card('spades', '5', 5)
    players[0].hand.append(spade5b)
    engine.step({"type": "play", "cards": [spade5b]})
    assert enemy2 in game_state.discard_pile.cards
    assert game_state.tavern_deck.peek_top() is enemy1  # tavern top unchanged


def test_hearts_before_diamonds_ordering(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    # Clubs enemy: neither hearts nor diamonds is immune, and no club is in
    # the combo so total_attack is undoubled.
    game_state.current_enemy = Enemy('clubs', 'J', 10, 'enemy', 'Jack of Clubs', 40, 0)
    # Hearts heals exactly total_value (6) cards from discard to tavern
    # bottom; pre-load 6 known healed cards and 4 known tavern cards.
    healed = [make_card('hearts', '7', 7) for _ in range(6)]
    game_state.discard_pile.cards = list(healed)
    tavern_cards = [make_card('clubs', str(v), v) for v in (2, 3, 4, 5)]
    game_state.tavern_deck.cards = list(tavern_cards)
    del players[0].hand[0:2]  # give A two draw slots (hand_limit 8)
    players[1].hand = []  # give B room to draw too
    h3 = make_card('hearts', '3', 3)
    d3 = make_card('diamonds', '3', 3)
    players[0].hand.extend([h3, d3])
    engine.step({"type": "play", "cards": [h3, d3]})
    # Hearts resolved FIRST: all 6 healed cards went to the tavern bottom.
    # Diamonds resolved SECOND: 6 draws came off the tavern TOP — the 4
    # pre-existing cards first, then 2 healed cards reached the players
    # (B's draw order proves the healed cards were drawn at all).
    assert game_state.discard_pile.cards == []
    assert players[0].hand[-2:] == [tavern_cards[0], tavern_cards[2]]  # A drew t1, t3
    b_hand = players[1].hand
    assert b_hand[:2] == [tavern_cards[1], tavern_cards[3]]  # B drew t2, t4 first
    assert len(b_hand) == 4
    assert all(c in healed for c in b_hand[2:])  # then 2 healed cards
    remaining = game_state.tavern_deck.cards
    assert len(remaining) == 4
    assert all(c in healed for c in remaining)


def test_jester_chooses_next_player(engine):
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    jester = make_card('joker', 'Jester', 0, 'jester')
    players[0].hand.append(jester)
    with patch("game.engine.show_jester_played"), \
            patch("game.engine.prompt_next_player", return_value=players[1]):
        obs, reward, done, info = engine.step({"type": "play", "cards": [jester]})
    assert info == {"result": "jester"}
    # The Jester's chosen player becomes the active player.
    assert game_state.get_active_player() is players[1]
    assert game_state.face_jester_played is True


def test_run_win_terminal_shows_win(engine):
    # FIX-7 regression: handle_play_action returning 'game_won' must break
    # run()'s inner loop cleanly so show_win() runs, instead of falling
    # through to current_enemy.effective_attack() with current_enemy None
    # (AttributeError). Construction per the Phase 2 pattern:
    # create_enemy_faces patched to a single 5-health enemy,
    # deal_initial_hands patched out, player 0 (hand_limit 1) holds exactly
    # one 5-value card — playing it defeats the only enemy, the castle is
    # empty, and run() terminates as a win.
    players = [Player("A", 1), Player("B", 8)]
    enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 5, 0)
    winning_card = make_card('hearts', '5', 5)
    players[0].hand.append(winning_card)
    show_win_mock = MagicMock()
    prompt = MagicMock(side_effect=[[winning_card]])
    with patch("game.engine.create_enemy_faces", return_value=([enemy], [], [])), \
            patch("game.engine.deal_initial_hands"), \
            patch("game.engine.prompt_card_indexes_to_play", prompt), \
            patch("game.engine.show_current_state"), patch("game.engine.show_player_hand"), \
            patch("game.engine.show_warning"), \
            patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"), \
            patch("game.engine.show_win", show_win_mock):
        engine.run(players)  # must return normally, no AttributeError
    game_state = engine.game_state
    assert show_win_mock.call_count == 1
    assert game_state.is_game_won() is True
    assert game_state.current_enemy is None
    assert game_state.castle_deck.is_empty()


def test_run_empty_hand_defense_after_defeat_terminates(engine):
    # Defect fix: after an Option-A defeat the SAME player continues; if that
    # player is now empty-handed and the revealed enemy's attack > 0, the
    # auto-yield moves to the defense phase — the defense branch must NOT
    # prompt for discards with an empty hand (unrecoverable input() loop).
    # Construction: create_enemy_faces patched to a 5-health enemy (attack
    # 10) followed by a spades Queen (attack 10); deal_initial_hands patched
    # out; player 0 (hand_limit 1) holds exactly one 5-card and defeats the
    # first enemy with it, then auto-yields into defense with an empty hand.
    players = [Player("A", 1), Player("B", 8)]
    first_enemy = Enemy('hearts', 'J', 10, 'enemy', 'Jack of Hearts', 5, 10)
    second_enemy = Enemy('spades', 'Q', 15, 'enemy', 'Queen of Spades', 30, 10)
    five = make_card('hearts', '5', 5)
    players[0].hand.append(five)
    prompt_play = MagicMock(side_effect=[[five]])
    prompt_discard = MagicMock()
    with patch("game.engine.create_enemy_faces", return_value=([first_enemy], [second_enemy], [])), \
            patch("game.engine.deal_initial_hands"), \
            patch("game.engine.prompt_card_indexes_to_play", prompt_play), \
            patch("game.engine.prompt_card_indexes_to_discard", prompt_discard), \
            patch("game.engine.show_current_state"), patch("game.engine.show_player_hand"), \
            patch("game.engine.show_warning"), \
            patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"), patch("game.engine.show_new_enemy"), \
            patch("game.engine.show_discard_phase"), \
            patch("game.engine.show_loss"):
        result = engine.run(players)
    game_state = engine.game_state
    # Only the defeat play was prompted — never a discard prompt.
    assert prompt_play.call_count == 1
    assert prompt_discard.call_count == 0
    # The defense branch terminated via the loss path (matching the gym).
    assert result[1] == -10
    assert result[2] is True
    assert result[3] == {"result": "player_defeated"}
    assert game_state.get_active_player() is players[0]  # loss on A's defense turn
    assert game_state.current_enemy is second_enemy  # defeat revealed it


def test_spade_pends_after_jester_defeat_new_battle(engine):
    # Defect fix: handle_enemy_defeat must reset enemy_immunity (alongside
    # face_jester_played) — otherwise, after a jester-assisted defeat, the
    # stale enemy_immunity=False keeps the pending-spades collection gate
    # from firing against the NEXT spades enemy (spades neither resolved
    # nor pended).
    players = [Player("A", 8), Player("B", 8)]
    engine.reset(players)
    game_state = engine.game_state
    first_enemy = Enemy('spades', 'J', 10, 'enemy', 'Jack of Spades', 5, 10)
    second_enemy = Enemy('spades', 'Q', 15, 'enemy', 'Queen of Spades', 30, 10)
    game_state.current_enemy = first_enemy
    game_state.castle_deck.cards = [second_enemy]  # deterministic next battle
    jester = make_card('joker', 'Jester', 0, 'jester')
    hearts5 = make_card('hearts', '5', 5)
    spade4 = make_card('spades', '4', 4)
    players[0].hand.extend([jester, hearts5, spade4])
    with patch("game.engine.show_jester_played"), \
            patch("game.engine.prompt_next_player", return_value=players[0]), \
            patch("game.engine.show_warning"), \
            patch("game.engine.show_played_cards"), patch("game.engine.show_damage_dealt"), \
            patch("game.engine.show_enemy_defeated"), patch("game.engine.show_new_enemy"):
        # 1) Jester vs the spades enemy: immunity negated.
        obs, reward, done, info = engine.step({"type": "play", "cards": [jester]})
        assert info == {"result": "jester"}
        assert game_state.enemy_immunity is False
        # 2) Defeat the first enemy (jester-assisted battle ends).
        obs, reward, done, info = engine.step({"type": "play", "cards": [hearts5]})
        assert info == {"result": "enemy_defeated"}
        assert game_state.current_enemy is second_enemy
        # The fix: the new battle starts with immunity ON again.
        assert game_state.enemy_immunity is True
        assert game_state.face_jester_played is False
        # 3) A spade vs the NEW spades enemy: blocked by immunity -> pends
        # (previously the stale immunity=False meant it was silently lost).
        obs, reward, done, info = engine.step({"type": "play", "cards": [spade4]})
    assert game_state.pending_spades == [spade4]
    assert second_enemy.shield == 0  # power blocked, card only pended
    assert info == {"result": "defense_required"}
