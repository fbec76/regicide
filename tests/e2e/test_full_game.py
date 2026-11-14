"""E2E regression contract: full games driven through Engine.step().

Every game is played with only *valid* actions (exact Card objects from the
active hand), so the suite exercises the rules engine — not the known bugs the
rule-fix phases address. Card conservation is asserted after every single step,
and each run is fully deterministic via a pinned seed.

Policy randomness uses a dedicated ``random.Random(seed)`` stream, separate
from the game's global-RNG stream, so policy choices never perturb deck order.

Two structural properties of the CURRENT engine were established empirically
while building this suite (~7,900 full games across 2/3/4 players, hand limits
5-8, and ten-plus policies). They matter for interpreting these tests:

1. A WIN terminal is unreachable from a fresh reset: 0 wins in ~7,900 games;
   the best policy ever defeated 5 of 12 enemies before ``player_defeated``.
   Hands only refill via the diamonds suit power, and defense thresholds
   (attack 10/15/20) outpace hand recovery. Consequently the win-path test
   (test_full_game_win_with_greedy_policy) uses a documented 1-enemy
   "shortened campaign" setup rather than a fresh-reset full game.

2. The game can get STUCK in a non-terminal infinite loop: when every player's
   hand is empty and the current enemy's shield >= its attack
   (effective_attack() == 0), each step is a yield that merely advances the
   turn, forever (observed: seed 296 with the random+largest policy — both
   hands empty, Jack of Clubs shield 13 vs attack 10, damage 19/20). The
   MAX_STEPS guard below catches this; FIX-4 ("stuck => loss") is the planned
   engine-side fix. This suite must keep passing before AND after that fix.
"""
import random

import pytest

from game.enemy import Enemy
from game.rules import is_valid_play


MAX_STEPS = 5000
FULL_CAMPAIGN_CARDS = 52  # tavern 40 + castle 12 (2 players, 0 Jesters)
SHORT_CAMPAIGN_CARDS = 41  # 1-enemy campaign: 11 upcoming castle enemies removed


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------

def _valid_combo_from_hand(hand, rng):
    """Return a random valid same-value combo from the hand, or None.

    Combos are 2-4 cards of the same value with sum <= 10 (rules.valid_combo);
    animals/jesters are excluded. Validity is double-checked with
    is_valid_play so the policy never submits an invalid play.
    """
    groups = {}
    for card in hand:
        if card.c_type in ("animal", "jester"):
            continue
        groups.setdefault(card.value, []).append(card)
    candidates = []
    for cards in groups.values():
        for size in (2, 3, 4):
            if len(cards) >= size:
                candidates.append(cards[:size])
    valid = [c for c in candidates if is_valid_play(c)]
    if not valid:
        return None
    return list(rng.choice(valid))


def _attack_action(rng, game_state, step_index):
    """Attack policy: yield when the hand is empty, occasionally a valid combo,
    otherwise a single random card (exact Card objects from the hand)."""
    active = game_state.get_active_player()
    hand = list(active.hand)
    if not hand:
        # Empty hand: the only action is a yield ({"cards": []} is accepted as
        # a yield by step_attack). If a rule fix makes that yield forbidden
        # while the hand is empty, the engine terminates the game instead
        # (stuck => loss, FIX-4), so this cannot loop forever — the MAX_STEPS
        # and consecutive-error guards in _drive_to_terminal catch it if it
        # ever does.
        return {"type": "play", "cards": []}
    if step_index % 5 == 4:
        combo = _valid_combo_from_hand(hand, rng)
        if combo is not None:
            return {"type": "play", "cards": combo}
    return {"type": "play", "cards": [rng.choice(hand)]}


def _greedy_attack_action(rng, game_state, step_index):
    """Attack policy: always play the highest-value single card from the hand."""
    active = game_state.get_active_player()
    hand = active.hand
    if not hand:
        return {"type": "play", "cards": []}
    return {"type": "play", "cards": [max(hand, key=lambda c: c.value)]}


def _defense_action(game_state):
    """Defense policy: greedy largest-first cards until the discard total
    covers the enemy's effective attack. If the whole hand cannot cover it,
    submit the full hand — defense_validation then fails first and the game
    terminates as a loss ({"result": "player_defeated"})."""
    active = game_state.get_active_player()
    needed = game_state.current_enemy.effective_attack()
    hand = sorted(active.hand, key=lambda c: c.value, reverse=True)
    if sum(c.value for c in hand) < needed:
        return {"type": "defend", "cards": hand}
    discards = []
    covered = 0
    for card in hand:
        if covered >= needed:
            break
        discards.append(card)
        covered += card.value
    return {"type": "defend", "cards": discards}


# ---------------------------------------------------------------------------
# Shared driver / invariants
# ---------------------------------------------------------------------------

def _drive_to_terminal(engine, rng, attack_policy, total_cards, expected_cards):
    """Step the engine until a terminal state, asserting card conservation
    after EVERY step. Returns (steps, reward, info)."""
    game_state = engine.game_state
    steps = 0
    consecutive_errors = 0
    while True:
        assert steps < MAX_STEPS, (
            f"Game did not terminate within {MAX_STEPS} steps — the engine "
            "reached the documented stuck state (all hands empty + fully "
            "shielded enemy => infinite non-terminal yield cycle); "
            "investigate. See module docstring, finding 2."
        )
        if engine.phase == "attack":
            action = attack_policy(rng, game_state, steps)
        else:
            action = _defense_action(game_state)
        obs, reward, done, info = engine.step(action)
        steps += 1
        assert total_cards(game_state) == expected_cards, (
            f"Card conservation broken after step {steps}: "
            f"{total_cards(game_state)} != {expected_cards}"
        )
        if "error" in info:
            consecutive_errors += 1
            # While any player holds cards a legal action always exists (any
            # single card is a valid play, rules.is_valid_play), so a long run
            # of rejected actions means the engine is stuck. Fail loudly
            # instead of looping to MAX_STEPS.
            assert consecutive_errors <= 100, (
                f"Engine appears stuck: rejected {consecutive_errors} actions "
                f"in a row (last error: {info['error']})"
            )
        else:
            consecutive_errors = 0
        if done:
            return steps, reward, info


def _assert_terminal_invariants(engine, info, total_cards, expected_cards):
    """Terminal-state contract pinned for the consolidation phase.

    NOTE on the planned bidirectional invariant "is_game_won() <=> castle deck
    empty <=> result == win": the literal "castle deck empty" half has a real
    corner case — a defense loss against the FINAL castle card leaves
    castle_deck empty while is_game_won() is False (current_enemy is still
    set). The invariant the engine actually maintains, and the one pinned
    here, is:
        is_game_won() <=> (castle deck empty AND current_enemy is None)
        is_game_won() <=> info["result"] == "win"
    """
    game_state = engine.game_state
    won = game_state.is_game_won()
    assert engine.done is True
    assert won == (info.get("result") == "win")
    assert won == (game_state.castle_deck.is_empty()
                   and game_state.current_enemy is None)
    assert total_cards(game_state) == expected_cards


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_game_seeded_policy_to_terminal(make_engine, total_cards):
    # Empirically pinned during Phase 6 with this exact policy: seed 3
    # terminates in 19 steps. Under the CURRENT engine balance the terminal is
    # a loss (player_defeated) — see module docstring finding 1 — but this
    # test accepts either terminal by design: the contract is "reaches a
    # terminal state through real step() calls with only valid actions, with
    # the terminal invariants and full card conservation intact", which must
    # hold before AND after the rule fixes even if trajectories change.
    seed = 3
    engine = make_engine(seed=seed)
    rng = random.Random(seed)
    steps, reward, info = _drive_to_terminal(
        engine, rng, _attack_action, total_cards, FULL_CAMPAIGN_CARDS)
    _assert_terminal_invariants(engine, info, total_cards, FULL_CAMPAIGN_CARDS)


def test_full_game_win_with_greedy_policy(make_engine, total_cards):
    # WIN-path contract, pinned via a 1-enemy "shortened campaign" (orchestrator
    # approved deviation): no fresh-reset game can reach a WIN in the current
    # engine (module docstring, finding 1 — 0 wins in ~7,900 games), so after
    # reset the 11 upcoming castle enemies are legitimately removed, leaving
    # the current enemy as the only one. The game is then driven through real
    # engine.step() calls with only valid actions to a genuine WIN terminal:
    # defeat => castle_deck.draw_one() returns None => is_game_won() =>
    # reward 10 - 0.01, done, {"result": "win"}.
    #
    # Seed provenance: seeds 0..50 were swept locally with this exact greedy
    # policy; ALL of them win the 1-enemy campaign (1-7 steps). Seed 0 is the
    # first — pinned (3 steps: attack -> defense -> win).
    #
    # Deviation from the fresh-reset contract: after setup the conservation
    # constant is 41 (52 - 11 removed enemy cards), asserted after every step.
    seed = 0
    engine = make_engine(seed=seed)
    rng = random.Random(seed)
    game_state = engine.game_state
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
    assert len(game_state.castle_deck.cards) == 11
    game_state.castle_deck.cards = []
    assert total_cards(game_state) == SHORT_CAMPAIGN_CARDS
    steps, reward, info = _drive_to_terminal(
        engine, rng, _greedy_attack_action, total_cards, SHORT_CAMPAIGN_CARDS)
    _assert_terminal_invariants(engine, info, total_cards, SHORT_CAMPAIGN_CARDS)
    # Full win-path pinning (what Phase 7 consolidation must preserve):
    assert info["result"] == "win"
    assert engine.done is True
    assert engine.game_state.is_game_won() is True
    assert engine.game_state.castle_deck.is_empty()
    assert engine.game_state.current_enemy is None
    assert engine.game_state.cards_against_enemy == []
    assert reward == pytest.approx(9.99, abs=1e-9)  # 10 win reward - 0.01 penalty


def test_full_game_scripted_loss_uncoverable_defense(make_engine, total_cards):
    engine = make_engine(seed=11)
    game_state = engine.game_state
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
    active = game_state.get_active_player()
    # Deterministic setup: swap the current enemy (1-for-1, so card
    # conservation is preserved) for a real Enemy whose attack (200) exceeds
    # any possible hand sum (max 8 cards x 10 = 80). No RNG dependence.
    game_state.current_enemy = Enemy(
        suit="spades", rank="K", value=20, c_type="enemy",
        display_name="King of Spades", health=40, attack=200)
    engine.phase = "defense"
    hand = list(active.hand)
    assert sum(c.value for c in hand) < game_state.current_enemy.effective_attack()
    obs, reward, done, info = engine.step({"type": "defend", "cards": hand})
    # step() adds its standard -0.01 dispatch penalty on top of the -10 loss
    # reward returned by step_defense => observable reward is -10.01.
    assert done is True
    assert reward == pytest.approx(-10.01, abs=1e-9)
    assert info["result"] == "player_defeated"
    # defense_validation fails before any mutation: hand and enemy untouched.
    assert active.hand == hand
    assert game_state.current_enemy.effective_attack() == 200
    assert game_state.is_game_won() is False
    assert total_cards(game_state) == FULL_CAMPAIGN_CARDS
