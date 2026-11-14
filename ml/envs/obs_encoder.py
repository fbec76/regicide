"""Observation encoder: GameState -> exactly-108-float player_count-invariant vector.

Pure consumer of GameState: reads raw attributes, never mutates anything.

Signature
---------
encode(gs: GameState, phase: str) -> np.ndarray

    gs    : live GameState (``engine.game_state``). The observation is encoded
            from the ACTIVE player's perspective, which is ``gs.players[0]``
            (the deque's leftmost entry) — derivable from gs alone.
    phase : engine phase string, ``"attack"`` or ``"defense"``. The phase lives
            on the Engine (``engine.phase``), NOT on GameState, so the env must
            pass it in explicitly. Validated at the boundary.
    returns: float32 ndarray, shape ``(OBS_DIM,)`` == ``(108,)``.

Canonical seat index: ``gs.yields_in_a_row`` is a dict built once at GameState
construction in original turn order and never reordered (rotation only touches
the players deque), so the active player's seat index is recovered as
``list(gs.yields_in_a_row).index(active.name)``. Requires unique player names
(guaranteed upstream by the CLI defaults and the web lobby).

Layout (dim offsets; total 108):

====  =========  ====  =======================================================
  0   own hand    49   7 slots x 7: is_empty, value/20, suit_onehot(4), is_jester
 49   enemy        8   suit_onehot(4), health_frac, damage_frac, shield/20,
                       jester_active/immune flag
 57   tavern      22   top-3 cards x 7-dim card slot (21) + remaining_norm (1)
 79   discard      8   per-suit count/40 (4) + per-suit value-sum/20 (4)
 87   castle      10   remaining/12 (1) + per-suit remaining/4 (4) +
                       max_health histogram, 5 buckets / 12 (5)
 97   opponents    7   player_count one-hot(3) [2,3,4], opponents total held
                       norm (1), jester-in-play norm (1), any-yielded flag (1),
                       own seat index norm (1)
104   meta         4   phase flag (1), yield_allowed (1),
                       cards_against_enemy count norm (1), value-sum/20 (1)
====  =========  ====  =======================================================

Normalization and interpretation choices (documented ambiguities):

- **R3 value denominator:** ALL card-value norms divide by 20 — own-hand
  value_norm, tavern card values, discard value-sum, meta value-sum, enemy
  shield. Number cards 2..10 -> 0.1..0.5; animals (value 1) -> 0.05; jesters
  (value 0) -> 0.0; royals placed in hand (tests / future modes) value
  10/15/20 -> 0.5/0.75/1.0. Box high stays 1.0.
- **Clipping:** value-sums and shield can exceed 20 over a long battle
  (hearts recycle discards back into the tavern), so those norms are clipped
  to 1.0 to keep every observation within [0, 1] Box bounds.
- **Empty slot encoding:** the plan's "empty slot = zeros" is applied to the
  card-attribute dims (value, suit one-hot, is_jester); the ``is_empty`` dim
  carries the distinguishing flag (1.0 for empty, 0.0 for occupied) — it
  would be a dead dimension otherwise. The same 7-dim slot encoder is used
  for hand padding and for absent tavern top-slots.
- **Jester-in-play norm:** chosen interpretation = jesters visible in the
  OPEN (tavern_deck.cards + discard_pile.cards), i.e. not held in a hand,
  divided by 2 (the maximum jester count, a 4-player game). At 2 players
  there are no jesters, so the norm is always 0.0 there.
- **Denominators:** hp fractions ÷ max_health; tavern/discard counts ÷ 40;
  castle remaining ÷ 12 (initial castle size), per-suit castle remaining ÷ 4
  (initial per-suit count); max_health histogram over bins
  [0,10) [10,20) [20,30) [30,40) [40,50] ÷ 12 (so the histogram sums to
  castle remaining_frac); opponents total held ÷ Σ opponent hand_limits;
  own seat index ÷ (player_count − 1); cards_against_enemy count ÷ Σ all
  hand_limits (clipped).
- **Phase flag:** 0.0 = attack, 1.0 = defense.
- **Jester-active/immune flag:** 0.0 while the current enemy is immune, 1.0
  once a Jester has been played against it (``gs.enemy_immunity`` False).
- **Won-game guard:** if ``gs.current_enemy`` is None (enemy defeated and no
  successor drawn yet) the enemy block encodes as all zeros.
- **Hand overflow:** hands cannot exceed hand_limit under engine rules; if a
  fixture ever supplies more than 7 cards, only the first 7 are encoded.
"""

from typing import Optional

import numpy as np

from game.card import Card
from game.game_state import GameState

OBS_DIM = 108
HAND_SLOTS = 7
TAVERN_TOP_SLOTS = 3
SUIT_ORDER = ("hearts", "diamonds", "clubs", "spades")
PLAYER_COUNTS = (2, 3, 4)
PHASES = ("attack", "defense")

VALUE_DENOM = 20.0   # R3: ALL card-value norms divide by 20
COUNT_DENOM = 40.0   # tavern/discard card counts divide by 40
CASTLE_SIZE = 12     # initial castle deck size (3 faces x 4 suits)
CASTLE_PER_SUIT = 4  # initial per-suit castle count
JESTER_DENOM = 2.0   # max jesters in any configuration (4 players)
HEALTH_BUCKET_WIDTH = 10.0
HEALTH_BUCKET_COUNT = 5  # bins [0,10) [10,20) [20,30) [30,40) [40,50]

# Block offsets — pinned by test_obs_total_dim_is_108.
_HAND_OFFSET = 0
_ENEMY_OFFSET = 49
_TAVERN_OFFSET = 57
_DISCARD_OFFSET = 79
_CASTLE_OFFSET = 87
_OPPONENTS_OFFSET = 97
_META_OFFSET = 104


def _clip01(x: float) -> float:
    return min(max(float(x), 0.0), 1.0)


def _suit_onehot(suit: str) -> np.ndarray:
    """4-dim one-hot over SUIT_ORDER; unknown suits (e.g. 'joker') -> zeros."""
    return np.array([1.0 if s == suit else 0.0 for s in SUIT_ORDER], dtype=np.float32)


def _card_slot(card: Optional[Card]) -> np.ndarray:
    """7-dim card-slot encoding shared by hand slots and tavern top slots.

    ``None`` (empty/absent slot) -> is_empty=1.0 with all attribute dims zero.
    """
    slot = np.zeros(7, dtype=np.float32)
    if card is None:
        slot[0] = 1.0
        return slot
    slot[1] = card.value / VALUE_DENOM
    slot[2:6] = _suit_onehot(card.suit)
    slot[6] = 1.0 if card.c_type == "jester" else 0.0
    return slot


def _hand_block(gs: GameState) -> np.ndarray:
    hand = list(gs.players[0].hand)[:HAND_SLOTS]
    slots = [_card_slot(c) for c in hand]
    slots += [_card_slot(None) for _ in range(HAND_SLOTS - len(slots))]
    return np.concatenate(slots)


def _enemy_block(gs: GameState) -> np.ndarray:
    block = np.zeros(8, dtype=np.float32)
    enemy = gs.current_enemy
    if enemy is None:
        return block  # game already won: no enemy on the table
    block[0:4] = _suit_onehot(enemy.suit)
    max_hp = enemy.max_health
    if max_hp > 0:
        block[4] = _clip01((max_hp - enemy.current_damage) / max_hp)
        block[5] = _clip01(enemy.current_damage / max_hp)
    block[6] = _clip01(enemy.shield / VALUE_DENOM)
    block[7] = 0.0 if gs.enemy_immunity else 1.0
    return block


def _tavern_block(gs: GameState) -> np.ndarray:
    cards = gs.tavern_deck.cards[:TAVERN_TOP_SLOTS]
    slots = [_card_slot(c) for c in cards]
    slots += [_card_slot(None) for _ in range(TAVERN_TOP_SLOTS - len(slots))]
    remaining = np.array([len(gs.tavern_deck.cards) / COUNT_DENOM], dtype=np.float32)
    return np.concatenate(slots + [remaining])


def _discard_block(gs: GameState) -> np.ndarray:
    suit_index = {s: i for i, s in enumerate(SUIT_ORDER)}
    counts = np.zeros(4, dtype=np.float32)
    sums = np.zeros(4, dtype=np.float32)
    for card in gs.discard_pile.cards:
        i = suit_index.get(card.suit)
        if i is not None:  # jester suit 'joker' is not one of the four suits
            counts[i] += 1.0
            sums[i] += card.value
    counts /= COUNT_DENOM
    sums = np.minimum(sums / VALUE_DENOM, 1.0)
    return np.concatenate([counts, sums])


def _castle_block(gs: GameState) -> np.ndarray:
    cards = gs.castle_deck.cards
    suit_index = {s: i for i, s in enumerate(SUIT_ORDER)}
    block = np.zeros(10, dtype=np.float32)
    block[0] = len(cards) / CASTLE_SIZE
    for card in cards:
        i = suit_index.get(card.suit)
        if i is not None:
            block[1 + i] += 1.0 / CASTLE_PER_SUIT
        health = getattr(card, "max_health", None)
        if health is not None:
            bucket = min(int(health // HEALTH_BUCKET_WIDTH), HEALTH_BUCKET_COUNT - 1)
            block[5 + bucket] += 1.0 / CASTLE_SIZE
    return block


def _seat_index(gs: GameState) -> int:
    """Active player's canonical seat index (0-based, original turn order).

    Derived from yields_in_a_row insertion order — see module docstring.
    """
    seat_order = list(gs.yields_in_a_row)
    name = gs.players[0].name
    if name not in seat_order:
        raise ValueError(
            f"active player {name!r} not found in yields_in_a_row seat order; "
            "player names must be unique"
        )
    return seat_order.index(name)


def _opponents_block(gs: GameState) -> np.ndarray:
    players = list(gs.players)
    active = players[0]
    opponents = players[1:]
    player_count = len(players)
    block = np.zeros(7, dtype=np.float32)
    block[PLAYER_COUNTS.index(player_count)] = 1.0
    held = sum(len(p.hand) for p in opponents)
    capacity = sum(p.hand_limit for p in opponents)
    if capacity > 0:
        block[3] = _clip01(held / capacity)
    open_cards = gs.tavern_deck.cards + gs.discard_pile.cards
    jesters_open = sum(1 for c in open_cards if c.c_type == "jester")
    block[4] = jesters_open / JESTER_DENOM
    block[5] = 1.0 if any(gs.yields_in_a_row.get(p.name, False) for p in opponents) else 0.0
    block[6] = _seat_index(gs) / (player_count - 1)
    return block


def _meta_block(gs: GameState, phase: str) -> np.ndarray:
    active = gs.players[0]
    block = np.zeros(4, dtype=np.float32)
    block[0] = 1.0 if phase == "defense" else 0.0
    block[1] = 1.0 if gs.can_yield(active) else 0.0
    total_capacity = sum(p.hand_limit for p in gs.players)
    if total_capacity > 0:
        block[2] = _clip01(len(gs.cards_against_enemy) / total_capacity)
    block[3] = _clip01(sum(c.value for c in gs.cards_against_enemy) / VALUE_DENOM)
    return block


def encode(gs: GameState, phase: str) -> np.ndarray:
    """Encode GameState + engine phase into a 108-float observation vector.

    Pure function: reads gs, returns a fresh array, mutates nothing.
    Raises ValueError on an unknown phase or unsupported player count.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown engine phase {phase!r}; expected one of {PHASES}")
    if len(gs.players) not in PLAYER_COUNTS:
        raise ValueError(
            f"unsupported player count {len(gs.players)}; expected one of {PLAYER_COUNTS}"
        )
    obs = np.concatenate(
        [
            _hand_block(gs),
            _enemy_block(gs),
            _tavern_block(gs),
            _discard_block(gs),
            _castle_block(gs),
            _opponents_block(gs),
            _meta_block(gs, phase),
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"observation layout broken: shape {obs.shape}, expected ({OBS_DIM},)")
    return obs
