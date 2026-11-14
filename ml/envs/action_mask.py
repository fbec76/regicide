"""Static Discrete(172) action-space layout for the Regicide RL env.

Pure numpy + game.rules — no sb3/torch (ADR-002 import-lint). This module is a
pure function library over (phase, hand, needed): it never mutates hands and
never touches the engine.

Index layout (static, hand-slot based)
--------------------------------------
    0           YIELD (play with no cards)
    1..7        singles: id ``1 + i`` plays ``hand[i]``
    8..22       same-rank combos (2-4 cards, same value, k*value <= 10;
                animals/jesters/enemy royals excluded) — rules.valid_combo
    23..40      animal pairs (animal + non-jester, or two animals) —
                rules.valid_animal_pair
    41..167     defense discard subsets: id ``40 + b`` (bitmask b in 1..127)
                discards the hand slots ``{i : bit i of b is set}`` (hand-slot
                bitmask encoding)
    168..171    reserved: Phase 6 jester next-player seat choice — always
                masked in Phases 1-5

Worst-case accounting (verified against the real rules functions)
-----------------------------------------------------------------
    singles     7  every card is a legal single (rules.is_valid_play)
    combos     15  per-hand max: 4x value-2 (C(4,2)+C(4,3)+C(4,4) = 11)
                   + 3x value-3 (C(3,2)+C(3,3) = 4); values >= 6 contribute 0
    pairs      18  max at 4 animals + 3 non-jester partners = 4*3 + C(4,2);
                   only 4 animal cards exist in the tavern deck, so a
                   7-card hand can never hold more than 4 animals
    attack     41  1 + 7 + 15 + 18 (block maxima, achieved by different hands)
    defense   127  2**7 - 1 non-empty subsets of a 7-card hand
    reserved    4
    N_ACTIONS 172

Per-state LIVE maximum = 34 attack actions (1 yield + 7 singles + 11 combos +
15 pairs), reached by the hand 4x value-2 + 3 animals. Brute-forced over all
deck-feasible 7-card compositions (animals <= 4, jesters <= 2, <= 4 copies per
value) and pinned by
``tests/ml/test_action_mask.py::test_worst_case_attack_hand_yields_34_actions``.

Static-slot mapping
-------------------
Blocks are capacity-bounded, not per-value fixed. Within a block, ids are
assigned by the canonical enumeration order of the CURRENT hand:

* combos: ascending value, then ascending sorted hand-index tuples
* pairs: ascending sorted hand-index tuples
* defense: ascending subset bitmask ``b`` in 1..127, mapped to id ``40 + b``
  (an injective encoding of all non-empty subsets onto ids 41..167; refines
  the plan's "sorted by size, then indices" note to a direct bitmask-to-id
  map)

``decode_action`` re-enumerates the same hand with the same rules functions,
so the mapping is a pure function of the hand and always round-trips.

YIELD gating (can_yield rule)
-----------------------------
In non-stuck attack states YIELD (id 0) is unmasked iff ``yield_allowed``
(the env passes ``GameState.can_yield(active_player)``). The engine refuses a
yield when every other player yielded on their last turn, so a masked-in
yield in such states would be rejected with {"error": "Yield forbidden"};
the env must therefore mask it out. In defense phase YIELD is always masked
(irrelevant there). The env callers pass ``can_yield`` unconditionally; the
parameter only affects the attack block.

R2 carve-out (stuck state)
--------------------------
When the active player has an empty hand AND yield is forbidden
(GameState.player_is_stuck) no legal action exists, and an all-False mask is
invalid for MaskablePPO. ``build_mask(..., stuck=True)`` therefore unmasks
YIELD ONLY (index 0) — regardless of ``yield_allowed``. The ENV handles the
loss itself (R2 is env-level, regicide_env): ``Engine.step`` runs
``_begin_turn`` BEFORE parsing the action and preempts a stuck turn with the
``player_defeated`` terminal, so the decoded yield
(``{"type": "play", "cards": []}``) is never consumed by the engine. This
unmasked yield does NOT pass ``can_yield`` semantics — a documented exception
for the Phase 1.6 mask-equivalence gate.

R1 (empty-hand / insufficient-hand defense auto-loss) is also env-level: a
defense mask for a hand that cannot cover ``needed`` is simply all-False on
the defense block (the env performs the auto-loss before presenting it).
"""
from __future__ import annotations

from itertools import combinations
from typing import TYPE_CHECKING, NamedTuple, Sequence

import numpy as np

from game.rules import valid_animal_pair, valid_combo

if TYPE_CHECKING:
    from game.card import Card

N_ACTIONS = 172
YIELD_ID = 0
SINGLE_START = 1  # ids 1..7: id 1+i plays hand[i]
SINGLE_COUNT = 7
COMBO_START = 8  # ids 8..22
COMBO_COUNT = 15
PAIR_START = 23  # ids 23..40
PAIR_COUNT = 18
ATTACK_SLOTS = 41  # ids 0..40 form the attack block
DEFENSE_BITMASK_OFFSET = 40  # defense id = 40 + bitmask, bitmask 1..127
DEFENSE_START = 41  # first defense id (bitmask 1)
DEFENSE_COUNT = 127  # 2**7 - 1 non-empty subsets of a <=7-card hand
DEFENSE_END_EXCLUSIVE = 168  # DEFENSE_START + DEFENSE_COUNT == RESERVED_START
RESERVED_START = 168  # ids 168..171: Phase 6 jester seat choice
MAX_HAND_SIZE = 7

_PHASES = ("attack", "defense")


class Action(NamedTuple):
    """One enumerated legal action; ``cards`` are exact Card objects from the hand."""

    id: int
    kind: str  # "yield" | "single" | "combo" | "pair" | "defend"
    cards: tuple  # tuple[Card, ...]


def _validate_hand(hand: Sequence["Card"]) -> None:
    if len(hand) > MAX_HAND_SIZE:
        raise ValueError(
            f"hand has {len(hand)} cards; static layout supports at most {MAX_HAND_SIZE}"
        )


def _validate_needed(needed: int | None) -> None:
    if not isinstance(needed, int) or needed < 0:
        raise ValueError(f"needed must be a non-negative int, got {needed!r}")


def _bitmask_indices(bitmask: int) -> list[int]:
    return [i for i in range(MAX_HAND_SIZE) if bitmask & (1 << i)]


def _enumerate_combos(hand: Sequence["Card"]) -> list[tuple[int, tuple]]:
    """Same-rank combos in canonical order -> [(action_id, cards)]."""
    found: list[tuple[int, tuple, tuple]] = []
    for k in (2, 3, 4):
        for idxs in combinations(range(len(hand)), k):
            cards = tuple(hand[i] for i in idxs)
            if valid_combo(list(cards)):
                found.append((cards[0].value, idxs, cards))
    found.sort(key=lambda entry: (entry[0], entry[1]))
    if len(found) > COMBO_COUNT:
        raise RuntimeError(
            f"{len(found)} combos exceed the static {COMBO_COUNT}-slot block"
        )
    return [(COMBO_START + i, cards) for i, (_, _, cards) in enumerate(found)]


def _enumerate_pairs(hand: Sequence["Card"]) -> list[tuple[int, tuple]]:
    """Animal pairs in canonical order -> [(action_id, cards)]."""
    found: list[tuple[tuple, tuple]] = []
    for idxs in combinations(range(len(hand)), 2):
        cards = tuple(hand[i] for i in idxs)
        if valid_animal_pair(list(cards)):
            found.append((idxs, cards))
    found.sort(key=lambda entry: entry[0])
    if len(found) > PAIR_COUNT:
        raise RuntimeError(f"{len(found)} pairs exceed the static {PAIR_COUNT}-slot block")
    return [(PAIR_START + i, cards) for i, (_, cards) in enumerate(found)]


def enumerate_attack_actions(hand: Sequence["Card"]) -> list[Action]:
    """All legal attack plays for ``hand``: yield, singles, combos, animal pairs.

    Enemy royals in hand are legal singles and valid animal partners
    (rules.is_valid_play / valid_animal_pair semantics); they are excluded
    from combos by valid_combo.
    """
    _validate_hand(hand)
    actions = [Action(YIELD_ID, "yield", ())]
    for i, card in enumerate(hand):
        actions.append(Action(SINGLE_START + i, "single", (card,)))
    actions.extend(Action(aid, "combo", cards) for aid, cards in _enumerate_combos(hand))
    actions.extend(Action(aid, "pair", cards) for aid, cards in _enumerate_pairs(hand))
    return actions


def enumerate_defense_actions(hand: Sequence["Card"], needed: int) -> list[Action]:
    """All non-empty hand subsets with value-sum >= needed (enemy royals count).

    Ids follow the hand-slot bitmask encoding: id 41+b for subset bitmask b,
    ascending. At most DEFENSE_COUNT (127) actions exist.
    """
    _validate_hand(hand)
    _validate_needed(needed)
    actions: list[Action] = []
    for bitmask in range(1, DEFENSE_COUNT + 1):
        idxs = _bitmask_indices(bitmask)
        if any(i >= len(hand) for i in idxs):
            continue
        cards = tuple(hand[i] for i in idxs)
        if sum(c.value for c in cards) >= needed:
            actions.append(Action(DEFENSE_BITMASK_OFFSET + bitmask, "defend", cards))
    return actions


def build_mask(
    phase: str,
    hand: Sequence["Card"],
    needed: int | None = None,
    stuck: bool = False,
    yield_allowed: bool = True,
) -> np.ndarray:
    """Boolean legality mask of shape (N_ACTIONS,); True = legal.

    Attack: YIELD iff ``yield_allowed`` (the env passes
    GameState.can_yield of the active player), plus every enumerated
    single/combo/pair. Defense: only subsets with value-sum >= needed; YIELD
    and all attack-play ids are always False there. Reserved ids are always
    False. ``stuck=True`` (attack only) unmasks YIELD ONLY, overriding
    ``yield_allowed`` — see the R2 carve-out in the module docstring.
    """
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}")
    _validate_hand(hand)
    mask = np.zeros(N_ACTIONS, dtype=bool)
    if phase == "defense":
        if stuck:
            raise ValueError("stuck is an attack-phase concept (R2); defense has no yield")
        _validate_needed(needed)
        for action in enumerate_defense_actions(hand, needed):
            mask[action.id] = True
        return mask
    if stuck:
        mask[YIELD_ID] = True
        return mask
    for action in enumerate_attack_actions(hand):
        mask[action.id] = True
    if not yield_allowed:
        mask[YIELD_ID] = False
    return mask


def decode_action(
    action_id: int, hand: Sequence["Card"], phase: str, needed: int | None
) -> dict:
    """Rebuild the engine action dict for ``action_id`` against the given hand.

    Returns {"type": "play"/"defend", "cards": [exact Card objects]}. YIELD
    decodes to {"type": "play", "cards": []}. Reserved ids (168..171) and ids
    that map to empty slots for this hand raise ValueError.
    """
    if phase not in _PHASES:
        raise ValueError(f"phase must be one of {_PHASES}, got {phase!r}")
    _validate_hand(hand)
    if not isinstance(action_id, int) or not 0 <= action_id < N_ACTIONS:
        raise ValueError(f"action_id {action_id!r} outside [0, {N_ACTIONS})")
    if action_id >= RESERVED_START:
        raise ValueError(f"action id {action_id} is reserved (Phase 6 jester seat choice)")
    if phase == "defense":
        _validate_needed(needed)
        if action_id < DEFENSE_START:
            raise ValueError(f"action id {action_id} is not a defense action")
        idxs = _bitmask_indices(action_id - DEFENSE_BITMASK_OFFSET)
        if any(i >= len(hand) for i in idxs):
            raise ValueError(f"action id {action_id} references slots beyond this hand")
        return {"type": "defend", "cards": [hand[i] for i in idxs]}
    if action_id == YIELD_ID:
        return {"type": "play", "cards": []}
    if SINGLE_START <= action_id < SINGLE_START + SINGLE_COUNT:
        i = action_id - SINGLE_START
        if i >= len(hand):
            raise ValueError(f"single slot {i} beyond this hand")
        return {"type": "play", "cards": [hand[i]]}
    for aid, cards in _enumerate_combos(hand) + _enumerate_pairs(hand):
        if aid == action_id:
            return {"type": "play", "cards": list(cards)}
    raise ValueError(f"action id {action_id} maps to an empty attack slot for this hand")
