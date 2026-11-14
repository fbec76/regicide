"""Tests for ml/envs/obs_encoder.py — the 108-float observation layout."""

import pickle

import numpy as np
import pytest

from game.card import Card
from game.enemy import Enemy
from game.engine import Engine
from game.player import Player

from ml.envs.obs_encoder import encode

# Offsets mirror the pinned layout in obs_encoder.py docstring.
_HAND = slice(0, 49)
_ENEMY = slice(49, 57)
_TAVERN = slice(57, 79)
_DISCARD = slice(79, 87)
_CASTLE = slice(87, 97)
_OPPONENTS = slice(97, 104)
_META = slice(104, 108)

HAND_LIMITS = {2: 7, 3: 6, 4: 5}


class DummyLogger:
    def info(self, msg): pass

    def warning(self, msg): pass

    def error(self, msg): pass

    def data(self, msg): pass


@pytest.fixture
def engine():
    return Engine(DummyLogger(), lambda name=None: DummyLogger())


def make_game(engine: Engine, player_count: int):
    """Real Engine.reset with the plan's hand limits (7/6/5 for 2/3/4 players)."""
    players = [Player(f"P{i}", HAND_LIMITS[player_count]) for i in range(player_count)]
    engine.reset(players)
    return engine.game_state


def make_card(suit, rank, value, c_type="number"):
    return Card(suit, rank, value, c_type, f"{rank} of {suit}")


def hand_slots(obs: np.ndarray) -> np.ndarray:
    return obs[_HAND].reshape(7, 7)


# --- layout -----------------------------------------------------------------

def test_obs_total_dim_is_108(engine):
    """Exactly 108 floats, float32, finite, within [0, 1] Box bounds.

    Note (R3): all card-value norms scale by denominator 20 — hand value_norm,
    tavern card values, discard value-sum, meta value-sum, enemy shield — so
    number cards land in [0.1, 0.5] and the dim count is unchanged at 108.
    """
    gs = make_game(engine, 2)
    obs = encode(gs, "attack")
    assert obs.shape == (108,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))
    assert obs.min() >= 0.0
    assert obs.max() <= 1.0


@pytest.mark.parametrize("player_count", [2, 3, 4])
def test_obs_shape_invariant_across_player_counts(engine, player_count):
    """Encoder output shape is identical at 2/3/4 players (Phase 7 semantics
    are later; the encoder must already be player_count-invariant)."""
    gs = make_game(engine, player_count)
    obs = encode(gs, "attack")
    assert obs.shape == (108,)


def test_encode_rejects_unknown_phase(engine):
    gs = make_game(engine, 2)
    with pytest.raises(ValueError):
        encode(gs, "nonsense")


# --- hand block -------------------------------------------------------------

def test_hand_padding_beyond_hand_size(engine):
    """A hand smaller than 7 cards pads trailing slots with is_empty=1.0 +
    zeroed card-attribute dims."""
    gs = make_game(engine, 2)
    gs.players[0].hand = gs.players[0].hand[:2]
    obs = encode(gs, "attack")
    slots = hand_slots(obs)
    assert np.all(slots[:2, 0] == 0.0)   # occupied slots: is_empty = 0
    assert np.all(slots[2:, 0] == 1.0)   # padded slots: is_empty = 1
    assert np.all(slots[2:, 1:] == 0.0)  # padded slots: attributes all zero


def test_empty_hand_slot_encoding(engine):
    """Fully empty hand: every slot is is_empty=1.0 with all-attribute zeros —
    clearly distinct from a populated hand."""
    gs = make_game(engine, 2)
    gs.players[0].hand = []
    empty_obs = encode(gs, "attack")
    slots = hand_slots(empty_obs)
    assert np.all(slots[:, 0] == 1.0)
    assert np.all(slots[:, 1:] == 0.0)

    populated_obs = encode(make_game(engine, 2), "attack")
    assert not np.array_equal(hand_slots(populated_obs), slots)


def test_jester_card_encoding(engine):
    """Jester: value 0 (0/20), 'joker' suit -> all-zero one-hot, is_jester=1."""
    gs = make_game(engine, 2)
    gs.players[0].hand = [Card("joker", "Jester", 0, "jester", "Jester")]
    obs = encode(gs, "attack")
    slot0 = hand_slots(obs)[0]
    assert slot0[0] == 0.0           # occupied
    assert slot0[1] == pytest.approx(0.0)   # value 0/20
    assert np.all(slot0[2:6] == 0.0)        # 'joker' not one of the four suits
    assert slot0[6] == 1.0           # is_jester


def test_royal_card_in_hand_within_obs_bounds(engine):
    """Royals placed in a hand: value 15 -> 0.75, value 20 -> 1.0 under ÷20;
    whole observation still within [0, 1]."""
    gs = make_game(engine, 2)
    gs.players[0].hand = [
        Enemy("hearts", "Q", 15, "enemy", "Queen of Hearts", health=30, attack=15),
        Enemy("clubs", "K", 20, "enemy", "King of Clubs", health=40, attack=20),
    ]
    obs = encode(gs, "attack")
    slots = hand_slots(obs)
    assert slots[0, 1] == pytest.approx(15 / 20)
    assert slots[1, 1] == pytest.approx(1.0)
    assert obs.max() <= 1.0


def test_suit_onehot_correct(engine):
    gs = make_game(engine, 2)
    gs.players[0].hand = [make_card("spades", "5", 5), make_card("hearts", "3", 3)]
    obs = encode(gs, "attack")
    slots = hand_slots(obs)
    # suit dims 2:6 in SUIT_ORDER = hearts, diamonds, clubs, spades
    assert np.array_equal(slots[0, 2:6], [0.0, 0.0, 0.0, 1.0])
    assert np.array_equal(slots[1, 2:6], [1.0, 0.0, 0.0, 0.0])
    assert slots[0, 1] == pytest.approx(5 / 20)
    assert slots[0, 6] == 0.0  # number card, not jester


def test_number_card_value_norm_is_20_scaled(engine):
    """R3 spot check: a ten encodes to 0.5 (10/20), a two to 0.1."""
    gs = make_game(engine, 2)
    gs.players[0].hand = [make_card("diamonds", "10", 10), make_card("clubs", "2", 2)]
    slots = hand_slots(encode(gs, "attack"))
    assert slots[0, 1] == pytest.approx(0.5)
    assert slots[1, 1] == pytest.approx(0.1)


# --- perspective ------------------------------------------------------------

def test_obs_from_active_player_perspective(engine):
    """Hand block always reflects players[0] (the active player), and the own
    seat index norm tracks the active player's canonical seat."""
    gs = make_game(engine, 2)
    first_card_value = gs.players[0].hand[0].value

    obs_before = encode(gs, "attack")
    assert hand_slots(obs_before)[0, 1] == pytest.approx(first_card_value / 20.0)
    assert obs_before[_OPPONENTS][-1] == pytest.approx(0.0)  # P0 sits at seat 0

    gs.advance_turn()  # P1 becomes active
    obs_after = encode(gs, "attack")
    new_first_value = gs.players[0].hand[0].value
    assert hand_slots(obs_after)[0, 1] == pytest.approx(new_first_value / 20.0)
    assert obs_after[_OPPONENTS][-1] == pytest.approx(1.0)  # P1 sits at seat 1


# --- enemy block ------------------------------------------------------------

def test_enemy_block_fractions_and_immunity_flag(engine):
    gs = make_game(engine, 2)
    enemy = gs.current_enemy
    enemy.current_damage = 10
    enemy.shield = 5
    obs = encode(gs, "attack")
    block = obs[_ENEMY]
    suit_pos = ("hearts", "diamonds", "clubs", "spades").index(enemy.suit)
    assert np.array_equal(block[0:4], np.eye(4, dtype=np.float32)[suit_pos])
    assert block[4] == pytest.approx((enemy.max_health - 10) / enemy.max_health)
    assert block[5] == pytest.approx(10 / enemy.max_health)
    assert block[6] == pytest.approx(5 / 20)
    assert block[7] == 0.0  # immune: no jester played yet

    gs.face_jester_played = True
    gs.update_enemy_immunity()
    assert encode(gs, "attack")[_ENEMY][7] == 1.0  # jester active, immunity off


# --- tavern block -----------------------------------------------------------

def test_tavern_block_top3_and_remaining_norm(engine):
    gs = make_game(engine, 2)
    obs = encode(gs, "attack")
    top3 = gs.tavern_deck.cards[:3]
    for i, card in enumerate(top3):
        slot = obs[_TAVERN][i * 7:(i + 1) * 7]
        assert slot[0] == 0.0
        assert slot[1] == pytest.approx(card.value / 20.0)
        assert slot[6] == (1.0 if card.c_type == "jester" else 0.0)
    assert obs[_TAVERN][-1] == pytest.approx(len(gs.tavern_deck.cards) / 40.0)


# --- discard block ----------------------------------------------------------

def test_discard_block_counts_and_value_sums(engine):
    gs = make_game(engine, 2)
    gs.discard_pile.cards = [
        make_card("hearts", "10", 10),
        make_card("hearts", "2", 2),
        make_card("spades", "3", 3),
    ]
    obs = encode(gs, "attack")
    block = obs[_DISCARD]
    # counts /40: hearts 2, diamonds 0, clubs 0, spades 1
    assert np.allclose(block[0:4], [2 / 40, 0.0, 0.0, 1 / 40])
    # value-sums /20: hearts 12, spades 3
    assert np.allclose(block[4:8], [12 / 20, 0.0, 0.0, 3 / 20])


def test_discard_value_sum_clipped_at_one(engine):
    """Value-sums exceeding the /20 denominator (hearts recycling) clip to 1.0
    so Box high stays 1.0."""
    gs = make_game(engine, 2)
    gs.discard_pile.cards = [make_card("clubs", "10", 10) for _ in range(3)]
    block = encode(gs, "attack")[_DISCARD]
    assert block[4 + 2] == 1.0  # clubs value-sum 30/20 -> clipped


# --- castle block -----------------------------------------------------------

def test_castle_block_remaining_suits_and_histogram(engine):
    gs = make_game(engine, 2)
    cards = gs.castle_deck.cards
    obs = encode(gs, "attack")
    block = obs[_CASTLE]
    assert block[0] == pytest.approx(len(cards) / 12)
    for i, suit in enumerate(("hearts", "diamonds", "clubs", "spades")):
        expected = sum(1 for c in cards if c.suit == suit) / 4
        assert block[1 + i] == pytest.approx(expected)
    # histogram bins [0,10) [10,20) [20,30) [30,40) [40,50] over max_health /12:
    # health 20 -> bucket 2, 30 -> bucket 3, 40 -> bucket 4 (bins 0-1 always 0)
    buckets = {20: 0.0, 30: 0.0, 40: 0.0}
    for c in cards:
        buckets[c.max_health] += 1 / 12
    assert np.allclose(block[5:10], [0.0, 0.0, buckets[20], buckets[30], buckets[40]])


# --- opponents block --------------------------------------------------------

def test_opponents_block_player_count_onehot_and_aggregates(engine):
    gs = make_game(engine, 3)
    opponents = list(gs.players)[1:]
    obs = encode(gs, "attack")
    block = obs[_OPPONENTS]
    assert np.array_equal(block[0:3], [0.0, 1.0, 0.0])  # 3 players
    held = sum(len(p.hand) for p in opponents)
    capacity = sum(p.hand_limit for p in opponents)
    assert block[3] == pytest.approx(held / capacity)


def test_opponents_block_any_yielded_flag(engine):
    gs = make_game(engine, 2)
    opponent = list(gs.players)[1]
    gs.yields_in_a_row[opponent.name] = True
    assert encode(gs, "attack")[_OPPONENTS][5] == 1.0
    gs.yields_in_a_row[opponent.name] = False
    assert encode(gs, "attack")[_OPPONENTS][5] == 0.0


# --- meta block -------------------------------------------------------------

def test_meta_block_phase_yield_and_cards_against_enemy(engine):
    gs = make_game(engine, 2)
    obs_attack = encode(gs, "attack")
    assert obs_attack[_META][0] == 0.0
    assert encode(gs, "defense")[_META][0] == 1.0
    assert obs_attack[_META][1] == (1.0 if gs.can_yield(gs.players[0]) else 0.0)

    gs.cards_against_enemy = [make_card("hearts", "10", 10)]
    total_capacity = sum(p.hand_limit for p in gs.players)
    obs = encode(gs, "attack")
    assert obs[_META][2] == pytest.approx(1 / total_capacity)
    assert obs[_META][3] == pytest.approx(10 / 20)


# --- purity -----------------------------------------------------------------

def test_encode_does_not_mutate_game_state(engine):
    """encode is a pure consumer: GameState is byte-identical before/after."""
    gs = make_game(engine, 2)
    before = pickle.dumps(gs)
    encode(gs, "attack")
    encode(gs, "defense")
    assert pickle.dumps(gs) == before
