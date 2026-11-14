"""Step 1.6 — MASK ≡ ENGINE VALIDATION EQUIVALENCE (Phase-1 exit gate).

Bidirectional proof that the static action mask (ml/envs/action_mask.py) and
the engine's own validation (game/rules.py + game/engine.py) can never
disagree, over real random play: seeded rollouts on the real RegicideGymEnv
at campaign lengths L ∈ {1, 3, 6} sample every pending state reached, and at
each sample every action index is classified against the engine's own rules.

Directions checked per sampled state
------------------------------------
- attack, forward:  every unmasked id decodes (decode_action) and passes
  rules.is_valid_play; yield id 0 is exempt from is_valid_play only in stuck
  states (R2 below) — in non-stuck states an unmasked id 0 must imply
  GameState.can_yield (bidirectional with the inverse check).
- attack, inverse:  every engine-legal play over the hand (all singles +
  all 2-4 card subsets via is_valid_play, i.e. valid_combo + animal pairs
  via valid_animal_pair) maps to an unmasked id; plus in non-stuck states
  mask[YIELD_ID] ⟺ GameState.can_yield (both directions).
- defense, forward: every unmasked id decodes to a subset whose value sum
  covers `needed` = current_enemy.effective_attack() — the engine's
  defense_validation semantics (sum(discards) >= needed).
- defense, inverse: every non-empty hand subset with value sum >= needed
  (2⁷−1 brute force) is unmasked.
- reserved ids 168..171 are masked in every sampled state.
- end-to-end: unmasked actions executed through an isolated clone of the
  sampled engine state never return {"error": ...}.

Carve-outs (plan review amendments R1/R2 — documented where they apply)
-----------------------------------------------------------------------
R1 (env auto-loss on unplayable defense): the env performs the auto-loss
itself (regicide_env._step_engine / step-entry guard) and NEVER presents an
insufficient-defense mask, so such states cannot appear among the samples;
the forward/inverse defense checks below are therefore vacuously complete
over presented defense states — nothing is excluded, the path simply never
reaches mask-based action selection.

R2 (stuck-state mask): when GameState.player_is_stuck the env presents a
yield-only mask. The decoded yield ({"type": "play", "cards": []}) does NOT
pass is_valid_play/can_yield semantics — Engine.step preempts a stuck turn
with the player_defeated terminal BEFORE parsing the action (engine._begin_turn).
Excluded from the is_valid_play check (test 1) and the executed pool
(test 5); stuck states still contribute to the inverse/reserved checks.

Yield id 0 in NON-stuck states (can_yield rule)
-----------------------------------------------
The env passes ``yield_allowed=GameState.can_yield(active)`` to build_mask,
so a non-stuck attack mask unmasks id 0 iff the engine would accept the
yield. The {"error": "Yield forbidden"} path (all other players yielded
last turn) is therefore unreachable through masked actions: an unmasked id 0
in a non-stuck state implies can_yield True, can_yield True implies id 0
unmasked, and id 0 is re-included in the executed pool (test 5) — it must
never produce {"error": "Yield forbidden"} anymore.

Engine-clone approach (test 5): copy.deepcopy(env.engine) fails — the silent
logger's NullHandler carries an unpicklable RLock, and the engine holds the
jester chooser closure. The cheaper correct approach is to deepcopy
engine.game_state at sample time (verified picklable-clean) and execute
through a fresh Engine shell built with the env's own silent-logger
factories (_make_engine_logger / _noop_get_logger) with a fixed jester
chooser (only consulted if the executed action plays a jester); phase/done
are copied explicitly. Each execution re-deepcopies the stored state so the
harness never mutates fixture data.

Determinism: every episode is reset(seed=s) (re-seeds global random once per
episode) and action selection uses a per-episode numpy default_rng, so the
sample set is identical on every run.
"""

import contextlib
import copy
import io
from itertools import combinations

import numpy as np
import pytest

from game.engine import Engine
from game.rules import is_valid_play

from ml.envs import action_mask
from ml.envs.action_mask import (
    DEFENSE_BITMASK_OFFSET,
    N_ACTIONS,
    RESERVED_START,
    SINGLE_START,
    YIELD_ID,
    decode_action,
    enumerate_attack_actions,
)
from ml.envs.regicide_env import (
    EnvConfig,
    RegicideGymEnv,
    _make_engine_logger,
    _noop_get_logger,
)

CAMPAIGN_LENGTHS = (1, 3, 6)
SAMPLE_QUOTA = {1: 220, 3: 240, 6: 220}
MAX_EPISODES_PER_LENGTH = 120
MIN_TOTAL_SAMPLES = 500
MIN_SAMPLES_PER_LENGTH = 150
MIN_EXECUTED_STATES = 500
ACTION_SEED_OFFSET = 7000
EXECUTION_SEED = 4242


# --------------------------------------------------------------------- #
# Sample collection (seeded random rollouts on the real env)
# --------------------------------------------------------------------- #
def _snapshot(env: RegicideGymEnv, seed: int) -> dict:
    """Immutable-enough snapshot of the CURRENT pending state.

    Validation runs immediately per state; only plain data (hand list copy,
    mask copy, ints, bools) plus a deep-copied game state for test 5 are
    stored — no live engine objects survive the snapshot.
    """
    game_state = env.engine.game_state
    phase = env.engine.phase
    active = game_state.get_active_player()
    return {
        "seed": seed,
        "phase": phase,
        "hand": list(active.hand),
        "needed": (
            game_state.current_enemy.effective_attack() if phase == "defense" else None
        ),
        "mask": env.current_mask().copy(),
        "stuck": (
            bool(game_state.player_is_stuck(active)) if phase == "attack" else False
        ),
        "can_yield": bool(game_state.can_yield(active)) if phase == "attack" else None,
        "gs_copy": copy.deepcopy(game_state),
    }


def _collect_samples(campaign_length: int, quota: int) -> list:
    """Random masked-action rollouts; snapshot every pending state reached.

    Each sample is one distinct pending-state visit (same-looking states in
    different episodes are distinct game states — deck/enemy context differs —
    so no dedupe across or within episodes).
    """
    env = RegicideGymEnv(EnvConfig(campaign_length=campaign_length))
    samples: list = []
    for seed in range(MAX_EPISODES_PER_LENGTH):
        if len(samples) >= quota:
            break
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed + ACTION_SEED_OFFSET)
        for _ in range(env.config.step_limit):
            if env.engine.done:
                break
            samples.append(_snapshot(env, seed))
            mask = env.current_mask()
            legal = np.flatnonzero(mask)
            action = int(rng.choice(legal))
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
    return samples


@pytest.fixture(scope="session")
def equivalence_samples() -> dict:
    """{campaign_length: [sample, ...]} — built once per session."""
    return {L: _collect_samples(L, SAMPLE_QUOTA[L]) for L in CAMPAIGN_LENGTHS}


# --------------------------------------------------------------------- #
# Violation collectors (shared by the named tests and the per-L gate)
# --------------------------------------------------------------------- #
def _violation(sample: dict, length: int, action_id, detail: str) -> dict:
    return {
        "L": length,
        "seed": sample["seed"],
        "phase": sample["phase"],
        "hand": [repr(card) for card in sample["hand"]],
        "action_id": action_id,
        "detail": detail,
    }


def _check_unmasked_attack_valid(length: int, states: list) -> list:
    """Forward attack direction: every unmasked id passes engine validation."""
    violations = []
    for sample in states:
        if sample["phase"] != "attack":
            continue
        unmasked = np.flatnonzero(sample["mask"])
        if sample["stuck"]:
            # R2 carve-out: a stuck mask must be yield-only; the decoded
            # yield never passes is_valid_play (engine preempts the turn).
            if list(unmasked) != [YIELD_ID]:
                violations.append(
                    _violation(sample, length, None, f"stuck mask unmasks {list(unmasked)}")
                )
            continue
        for raw_id in unmasked:
            action_id = int(raw_id)
            if action_id == YIELD_ID:
                # Yield is not an is_valid_play play; its engine gate is
                # GameState.can_yield — in non-stuck states an unmasked id 0
                # must imply can_yield True (R2 exempts stuck masks only).
                if not sample["can_yield"]:
                    violations.append(
                        _violation(
                            sample, length, YIELD_ID, "yield unmasked while can_yield False"
                        )
                    )
                continue
            decoded = decode_action(action_id, sample["hand"], "attack", None)
            if not is_valid_play(list(decoded["cards"])):
                violations.append(
                    _violation(
                        sample, length, action_id, "unmasked play fails is_valid_play"
                    )
                )
    return violations


def _attack_id_by_cards(hand: list) -> dict:
    """frozenset(card identities) -> action id over the enumerated attacks."""
    return {
        frozenset(action.cards): action.id for action in enumerate_attack_actions(hand)
    }


def _check_engine_legal_attack_unmasked(length: int, states: list) -> list:
    """Inverse attack direction: every engine-legal play must be unmasked."""
    violations = []
    for sample in states:
        if sample["phase"] != "attack":
            continue
        hand, mask = sample["hand"], sample["mask"]
        id_by_cards = _attack_id_by_cards(hand)
        # All singles: is_valid_play([card]) is always True for the engine.
        for i, card in enumerate(hand):
            if not is_valid_play([card]) or not mask[SINGLE_START + i]:
                violations.append(
                    _violation(sample, length, SINGLE_START + i, "legal single masked out")
                )
        # All 2-4 card subsets via the REAL rules (combos + animal pairs).
        for k in (2, 3, 4):
            for idxs in combinations(range(len(hand)), k):
                cards = [hand[i] for i in idxs]
                if not is_valid_play(cards):
                    continue
                mapped = id_by_cards.get(frozenset(cards))
                if mapped is None or not mask[mapped]:
                    violations.append(
                        _violation(
                            sample,
                            length,
                            mapped,
                            f"engine-legal play {[repr(c) for c in cards]} masked out",
                        )
                    )
        # Non-stuck yield equivalence (both directions): mask[YIELD_ID] must
        # equal GameState.can_yield. Stuck states are R2-exempt (yield-only
        # mask is presented even though can_yield is False there).
        if not sample["stuck"]:
            if sample["can_yield"] and not mask[YIELD_ID]:
                violations.append(
                    _violation(sample, length, YIELD_ID, "can_yield yield masked out")
                )
            if not sample["can_yield"] and mask[YIELD_ID]:
                violations.append(
                    _violation(sample, length, YIELD_ID, "yield unmasked while can_yield False")
                )
    return violations


def _check_unmasked_defense_valid(length: int, states: list) -> list:
    """Forward defense direction: unmasked ids decode to covering subsets."""
    violations = []
    for sample in states:
        if sample["phase"] != "defense":
            continue
        needed, hand, mask = sample["needed"], sample["hand"], sample["mask"]
        if not mask.any():
            # R1 interplay: an insufficient-defense state is auto-lost by the
            # env and never presented, so a presented mask must be non-empty.
            violations.append(
                _violation(sample, length, None, "presented defense mask is all-False")
            )
            continue
        for raw_id in np.flatnonzero(mask):
            action_id = int(raw_id)
            decoded = decode_action(action_id, hand, "defense", needed)
            if sum(card.value for card in decoded["cards"]) < needed:
                violations.append(
                    _violation(
                        sample,
                        length,
                        action_id,
                        f"unmasked defense subset covers {sum(c.value for c in decoded['cards'])} < needed {needed}",
                    )
                )
    return violations


def _check_engine_legal_defense_unmasked(length: int, states: list) -> list:
    """Inverse defense direction: every covering subset must be unmasked."""
    violations = []
    for sample in states:
        if sample["phase"] != "defense":
            continue
        needed, hand, mask = sample["needed"], sample["hand"], sample["mask"]
        for bitmask in range(1, 1 << len(hand)):
            cards = [hand[i] for i in range(len(hand)) if bitmask >> i & 1]
            if sum(card.value for card in cards) < needed:
                continue
            action_id = DEFENSE_BITMASK_OFFSET + bitmask
            if not mask[action_id]:
                violations.append(
                    _violation(
                        sample,
                        length,
                        action_id,
                        f"covering subset {[repr(c) for c in cards]} masked out (needed {needed})",
                    )
                )
    return violations


def _check_reserved_slots(length: int, states: list) -> list:
    violations = []
    for sample in states:
        if sample["mask"][RESERVED_START:N_ACTIONS].any():
            violations.append(
                _violation(sample, length, None, "reserved slots 168..171 unmasked")
            )
    return violations


ALL_DIRECTION_CHECKS = (
    _check_unmasked_attack_valid,
    _check_engine_legal_attack_unmasked,
    _check_unmasked_defense_valid,
    _check_engine_legal_defense_unmasked,
    _check_reserved_slots,
)


def _counts(equivalence_samples: dict) -> str:
    lines = []
    for length in CAMPAIGN_LENGTHS:
        states = equivalence_samples[length]
        attack = sum(1 for s in states if s["phase"] == "attack")
        defense = sum(1 for s in states if s["phase"] == "defense")
        stuck = sum(1 for s in states if s["phase"] == "attack" and s["stuck"])
        forbidden_yield = sum(
            1
            for s in states
            if s["phase"] == "attack" and not s["stuck"] and not s["can_yield"]
        )
        lines.append(
            f"L={length}: total={len(states)} attack={attack} defense={defense} "
            f"stuck={stuck} forbidden_yield={forbidden_yield}"
        )
    total = sum(len(states) for states in equivalence_samples.values())
    return f"samples total={total} | " + " | ".join(lines)


def _fail(violations: list, equivalence_samples: dict) -> str:
    shown = "; ".join(
        f"L={v['L']} seed={v['seed']} {v['phase']} id={v['action_id']}: {v['detail']}"
        for v in violations[:5]
    )
    return f"{len(violations)} mask/engine violations ({_counts(equivalence_samples)}). First: {shown}"


# --------------------------------------------------------------------- #
# The gate tests
# --------------------------------------------------------------------- #
def test_every_unmasked_attack_action_passes_engine_validation(equivalence_samples):
    """Forward attack: every unmasked attack id decodes and is engine-legal."""
    violations = []
    for length in CAMPAIGN_LENGTHS:
        violations.extend(
            _check_unmasked_attack_valid(length, equivalence_samples[length])
        )
    assert not violations, _fail(violations, equivalence_samples)


def test_every_engine_legal_attack_action_is_unmasked(equivalence_samples):
    """Inverse attack: every engine-legal play over the hand is unmasked."""
    violations = []
    for length in CAMPAIGN_LENGTHS:
        violations.extend(
            _check_engine_legal_attack_unmasked(length, equivalence_samples[length])
        )
    assert not violations, _fail(violations, equivalence_samples)


def test_every_unmasked_defense_subset_passes_defense_validation(equivalence_samples):
    """Forward defense: unmasked ids decode to subsets with sum >= needed."""
    violations = []
    for length in CAMPAIGN_LENGTHS:
        violations.extend(
            _check_unmasked_defense_valid(length, equivalence_samples[length])
        )
    assert not violations, _fail(violations, equivalence_samples)


def test_every_engine_legal_defense_subset_is_unmasked(equivalence_samples):
    """Inverse defense: every subset with sum >= needed (2⁷−1) is unmasked."""
    violations = []
    for length in CAMPAIGN_LENGTHS:
        violations.extend(
            _check_engine_legal_defense_unmasked(length, equivalence_samples[length])
        )
    assert not violations, _fail(violations, equivalence_samples)


def test_executed_unmasked_actions_never_error_over_500_states(equivalence_samples):
    """End-to-end: unmasked actions executed through an isolated engine-state
    clone never return {"error": ...} — the engine never rejects what the
    mask called legal.

    Exclusions (documented): stuck attack states (R2 — the engine preempts
    the turn before parsing). Every other unmasked id is eligible — including
    yield id 0 (the env masks it out whenever can_yield is False, so an
    executed yield is always engine-accepted); one action per sampled state
    is executed (deterministically chosen).
    """
    rng = np.random.default_rng(EXECUTION_SEED)
    executed = 0
    errors = []
    for length in CAMPAIGN_LENGTHS:
        for sample in equivalence_samples[length]:
            if sample["phase"] == "attack" and sample["stuck"]:
                continue  # R2 carve-out
            pool = [int(raw_id) for raw_id in np.flatnonzero(sample["mask"])]
            if not pool:
                continue
            action_id = int(rng.choice(pool))
            game_state = copy.deepcopy(sample["gs_copy"])
            clone = Engine(
                _make_engine_logger(),
                _noop_get_logger,
                next_player_chooser=lambda players: players[0],
            )
            clone.game_state = game_state
            clone.phase = sample["phase"]
            clone.done = False
            hand = game_state.get_active_player().hand
            action = decode_action(action_id, hand, sample["phase"], sample["needed"])
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, _, info = clone.step(action)
            executed += 1
            if "error" in info:
                errors.append(
                    _violation(sample, length, action_id, f"engine error {info['error']!r}")
                )
    assert executed >= MIN_EXECUTED_STATES, (
        f"only {executed} states executed (< {MIN_EXECUTED_STATES}); "
        f"{_counts(equivalence_samples)}"
    )
    assert not errors, _fail(errors, equivalence_samples)


def test_reserved_slots_never_unmasked(equivalence_samples):
    """Ids 168..171 (Phase 6 jester seat choice) are masked in every state."""
    violations = []
    for length in CAMPAIGN_LENGTHS:
        violations.extend(_check_reserved_slots(length, equivalence_samples[length]))
    assert not violations, _fail(violations, equivalence_samples)


def test_equivalence_holds_across_campaign_lengths(equivalence_samples):
    """The gate holds per campaign length; per-L counts localize failures."""
    total = sum(len(states) for states in equivalence_samples.values())
    assert total >= MIN_TOTAL_SAMPLES, (
        f"only {total} samples collected (< {MIN_TOTAL_SAMPLES}); "
        f"{_counts(equivalence_samples)}"
    )
    for length in CAMPAIGN_LENGTHS:
        states = equivalence_samples[length]
        assert len(states) >= MIN_SAMPLES_PER_LENGTH, (
            f"L={length} has only {len(states)} samples "
            f"(< {MIN_SAMPLES_PER_LENGTH}); {_counts(equivalence_samples)}"
        )
        length_violations = []
        for check in ALL_DIRECTION_CHECKS:
            length_violations.extend(check(length, states))
        assert not length_violations, (
            f"L={length} violates mask/engine equivalence: "
            f"{_fail(length_violations, equivalence_samples)}"
        )
    print("\nStep 1.6 equivalence gate:", _counts(equivalence_samples))
