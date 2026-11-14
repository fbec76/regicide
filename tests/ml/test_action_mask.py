"""Tests for ml/envs/action_mask.py — static Discrete(172) layout.

Uses real Card objects and the real rules functions (mirrors tests/test_rules.py
make_card pattern); no mocks.
"""
from itertools import combinations

import numpy as np
import pytest

from game.card import Card
from game.rules import is_valid_play, valid_animal_pair, valid_combo
from ml.envs.action_mask import (
    COMBO_START,
    DEFENSE_BITMASK_OFFSET,
    DEFENSE_COUNT,
    DEFENSE_END_EXCLUSIVE,
    DEFENSE_START,
    N_ACTIONS,
    PAIR_START,
    RESERVED_START,
    SINGLE_START,
    YIELD_ID,
    build_mask,
    decode_action,
    enumerate_attack_actions,
    enumerate_defense_actions,
)


def make_card(suit, rank, value, c_type):
    return Card(suit, rank, value, c_type, f"{rank} of {suit}")


def value_2_hand():
    """Worst-case attack fixture per plan: 4x value-2 + 3 animals."""
    return (
        [make_card(s, "2", 2, "number") for s in ("hearts", "spades", "clubs", "diamonds")]
        + [make_card(s, "A", 1, "animal") for s in ("hearts", "spades", "clubs")]
    )


def test_n_actions_is_172():
    assert N_ACTIONS == 172
    mask = build_mask("attack", value_2_hand())
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (172,)
    assert mask.dtype == bool
    assert len(build_mask("defense", value_2_hand(), needed=5)) == 172


def test_attack_enumeration_within_static_slots():
    hands = [
        value_2_hand(),
        # static combo max: 4x value-2 + 3x value-3 -> 11 + 4 = 15 combos
        [make_card(s, "2", 2, "number") for s in ("hearts", "spades", "clubs", "diamonds")]
        + [make_card(s, "3", 3, "number") for s in ("hearts", "spades", "clubs")],
        # static pair max: 4 animals + 3 non-jester partners -> 18 pairs
        [make_card(s, "A", 1, "animal") for s in ("hearts", "spades", "clubs", "diamonds")]
        + [make_card(s, "2", 2, "number") for s in ("hearts", "spades", "clubs")],
        # mixed: jester + enemy royal + numbers
        [
            make_card("joker", "Jester", 0, "jester"),
            make_card("spades", "Q", 15, "enemy"),
            make_card("hearts", "2", 2, "number"),
            make_card("spades", "2", 2, "number"),
            make_card("hearts", "A", 1, "animal"),
            make_card("hearts", "5", 5, "number"),
            make_card("clubs", "10", 10, "number"),
        ],
    ]
    for hand in hands:
        actions = enumerate_attack_actions(hand)
        kinds = [a.kind for a in actions]
        assert kinds.count("yield") == 1
        assert kinds.count("single") == len(hand) <= 7
        assert kinds.count("combo") <= 15
        assert kinds.count("pair") <= 18
        for action in actions:
            if action.kind == "yield":
                assert action.id == YIELD_ID
            elif action.kind == "single":
                assert SINGLE_START <= action.id < SINGLE_START + 7
            elif action.kind == "combo":
                assert COMBO_START <= action.id < PAIR_START
            elif action.kind == "pair":
                assert PAIR_START <= action.id < DEFENSE_START
            else:
                pytest.fail(f"unexpected kind {action.kind}")


def test_worst_case_attack_hand_yields_34_actions():
    """Live attack max: 1 yield + 7 singles + 11 combos + 15 pairs = 34.

    Brute-force cross-check with the REAL rules functions on the plan fixture
    (4x value-2 + 3 animals); the plan's 34 is confirmed, not assumed.
    """
    hand = value_2_hand()
    actions = enumerate_attack_actions(hand)
    # independent recount straight from the rules functions
    combos = {
        idxs
        for k in (2, 3, 4)
        for idxs in combinations(range(7), k)
        if valid_combo([hand[i] for i in idxs])
    }
    pairs = {
        idxs
        for idxs in combinations(range(7), 2)
        if valid_animal_pair([hand[i] for i in idxs])
    }
    assert len(combos) == 11
    assert len(pairs) == 15  # 3 animals x 4 others + C(3,2)
    assert len(actions) == 1 + 7 + len(combos) + len(pairs)
    assert len(actions) == 34


def test_defense_enumeration_capped_at_127():
    hand = [make_card(s, str(v), v, "number") for s, v in
            (("hearts", 2), ("spades", 2), ("clubs", 2), ("diamonds", 2),
             ("hearts", 3), ("spades", 3), ("clubs", 3))]
    # every value >= 1 and needed == 1 -> all 2**7 - 1 non-empty subsets qualify
    actions = enumerate_defense_actions(hand, needed=1)
    assert len(actions) == DEFENSE_COUNT == 127
    assert all(DEFENSE_START <= a.id < DEFENSE_END_EXCLUSIVE for a in actions)
    assert sorted(a.id for a in actions) == list(range(DEFENSE_START, DEFENSE_END_EXCLUSIVE))
    # no hand can enumerate more than the 127 static slots
    assert len(enumerate_defense_actions(hand, needed=100)) <= 127


def test_defense_mask_subset_sum_ge_needed():
    hand = [
        make_card("hearts", "5", 5, "number"),
        make_card("spades", "3", 3, "number"),
        make_card("clubs", "2", 2, "number"),
        make_card("diamonds", "2", 2, "number"),
        make_card("hearts", "A", 1, "animal"),
        make_card("spades", "A", 1, "animal"),
        make_card("spades", "Q", 15, "enemy"),
    ]
    needed = 15
    mask = build_mask("defense", hand, needed=needed)
    # expected: every in-hand non-empty subset with value-sum >= needed unmasked
    expected_ids = set()
    for bitmask in range(1, 128):
        idxs = [i for i in range(7) if bitmask & (1 << i)]
        subset_sum = sum(hand[i].value for i in idxs)
        if subset_sum >= needed:
            expected_ids.add(DEFENSE_BITMASK_OFFSET + bitmask)
    unmasked_ids = {i for i in range(DEFENSE_START, DEFENSE_END_EXCLUSIVE) if mask[i]}
    assert unmasked_ids == expected_ids
    # queen alone (15) covers needed; small subsets do not
    assert mask[DEFENSE_BITMASK_OFFSET + (1 << 6)]  # queen-only subset
    assert not mask[DEFENSE_BITMASK_OFFSET + (1 << 4)]  # animal alone: 1 < 15
    # attack-phase play ids must never be unmasked in defense
    assert not mask[:DEFENSE_START].any()


def test_yield_masked_when_not_allowed():
    """can_yield False (all other players yielded last turn) -> yield masked."""
    hand = value_2_hand()
    defense_mask = build_mask("defense", hand, needed=10)
    assert not defense_mask[YIELD_ID]
    attack_mask = build_mask("attack", hand, yield_allowed=False)
    assert not attack_mask[YIELD_ID]
    # every other attack action stays unmasked
    for action in enumerate_attack_actions(hand):
        if action.kind != "yield":
            assert attack_mask[action.id]
    attack_mask = build_mask("attack", hand, yield_allowed=True)
    assert attack_mask[YIELD_ID]
    # yield decodes as an empty play
    assert decode_action(YIELD_ID, hand, "attack", None) == {"type": "play", "cards": []}


def test_stuck_yields_even_when_can_yield_false():
    """R2 precedence: stuck overrides yield_allowed=False -> YIELD unmasked."""
    mask = build_mask("attack", [], stuck=True, yield_allowed=False)
    assert mask[YIELD_ID]
    assert mask.sum() == 1


def test_reserved_slots_always_masked():
    hand = value_2_hand()
    for phase, kwargs in (("attack", {}), ("defense", {"needed": 10})):
        mask = build_mask(phase, hand, **kwargs)
        assert not mask[RESERVED_START:N_ACTIONS].any()
    for action_id in range(RESERVED_START, N_ACTIONS):
        with pytest.raises(ValueError):
            decode_action(action_id, hand, "attack", None)
        with pytest.raises(ValueError):
            decode_action(action_id, hand, "defense", 10)


def test_decode_action_roundtrip():
    hand = value_2_hand()
    # attack: every enumerated (non-yield) action decodes to its exact cards
    # and passes the real engine-side validation function
    for action in enumerate_attack_actions(hand):
        decoded = decode_action(action.id, hand, "attack", None)
        assert decoded["type"] == "play"
        assert tuple(decoded["cards"]) == action.cards
        if action.kind != "yield":
            assert is_valid_play(decoded["cards"])
    # defense: every enumerated action decodes to its exact subset and covers needed
    needed = 10
    for action in enumerate_defense_actions(hand, needed=needed):
        decoded = decode_action(action.id, hand, "defense", needed)
        assert decoded["type"] == "defend"
        assert tuple(decoded["cards"]) == action.cards
        assert sum(c.value for c in decoded["cards"]) >= needed
        assert all(any(c is h for h in hand) for c in decoded["cards"])


def test_jester_single_is_single_action():
    jester = make_card("joker", "Jester", 0, "jester")
    hand = [
        jester,
        make_card("hearts", "2", 2, "number"),
        make_card("spades", "2", 2, "number"),
        make_card("hearts", "A", 1, "animal"),
    ]
    actions = enumerate_attack_actions(hand)
    jester_actions = [a for a in actions if any(c is jester for c in a.cards)]
    assert len(jester_actions) == 1
    assert jester_actions[0].kind == "single"
    assert jester_actions[0].cards == (jester,)
    decoded = decode_action(jester_actions[0].id, hand, "attack", None)
    assert decoded == {"type": "play", "cards": [jester]}
    # jesters cannot be in combos or animal pairs
    assert not any(a.kind in ("combo", "pair") for a in jester_actions)


def test_enemy_royal_in_hand():
    queen = make_card("spades", "Q", 15, "enemy")
    hand = [
        make_card("hearts", "A", 1, "animal"),
        queen,
        make_card("hearts", "2", 2, "number"),
    ]
    # real rules semantics: legal single, valid animal partner, no combos
    assert is_valid_play([queen])
    assert valid_animal_pair([hand[0], queen])
    assert not valid_combo([queen, make_card("spades", "Q", 15, "enemy")])
    actions = enumerate_attack_actions(hand)
    singles = [a for a in actions if a.kind == "single"]
    assert any(a.cards == (queen,) for a in singles)
    pairs = [a for a in actions if a.kind == "pair"]
    assert any(tuple(a.cards) == (hand[0], queen) for a in pairs)
    assert not any(a.kind == "combo" and queen in a.cards for a in actions)
    # queen's value 15 counts toward the defense sum
    mask = build_mask("defense", hand, needed=15)
    queen_id = DEFENSE_BITMASK_OFFSET + (1 << 1)  # queen at hand slot 1
    assert mask[queen_id]  # queen-only subset: 15 >= 15
    assert not mask[DEFENSE_BITMASK_OFFSET + (1 << 2)]  # two alone: 2 < 15
    decoded = decode_action(queen_id, hand, "defense", 15)
    assert decoded == {"type": "defend", "cards": [queen]}


def test_stuck_mask_unmasks_yield_only():
    mask = build_mask("attack", [], stuck=True)
    assert mask[YIELD_ID]
    assert mask.sum() == 1
    assert decode_action(YIELD_ID, [], "attack", None) == {"type": "play", "cards": []}
    with pytest.raises(ValueError):
        build_mask("defense", [], needed=5, stuck=True)


def test_invalid_inputs_raise():
    hand = value_2_hand()
    with pytest.raises(ValueError):
        build_mask("bogus", hand)
    with pytest.raises(ValueError):
        build_mask("attack", hand + [make_card("hearts", "3", 3, "number")])  # 8 cards
    with pytest.raises(ValueError):
        build_mask("defense", hand, needed=None)
    with pytest.raises(ValueError):
        decode_action(-1, hand, "attack", None)
    with pytest.raises(ValueError):
        decode_action(N_ACTIONS, hand, "attack", None)
    with pytest.raises(ValueError):
        decode_action(SINGLE_START + 6, hand[:2], "attack", None)  # empty slot
    with pytest.raises(ValueError):
        decode_action(DEFENSE_START, hand, "attack", None)  # defense id in attack
