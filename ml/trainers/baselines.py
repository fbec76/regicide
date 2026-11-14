"""Scripted baseline agents (Step 2.1).

All agents choose ONLY from the env's legality mask (``info["mask"]``) — they
are legal by construction and never emit an invalid action. Unlike the learned
policy (which sees only the 108-float observation), baselines ARE game-state
aware: they read ``env.engine.game_state`` to decode actions into cards and to
score plays. This is intentional — scripted baselines are meant to be strong,
legible reference points for the policy to beat (plan Step 2.1).

Decision rules
--------------
RandomAgent      uniform over unmasked action ids.
GreedyHighest    attack: among unmasked non-yield plays pick max total card
                 value (tie -> random); yield ONLY if it is the single
                 unmasked action (the R2 stuck-state carve-out). defense:
                 smallest subset meeting ``needed`` — min total value with
                 total >= needed, tie -> fewest cards (minimal overspend).
ShieldGreedy     same as GreedyHighest, but on attack it restricts candidates
                 to spade-containing plays while the enemy is unshielded AND
                 a spade would actually fire (spade suit power raises the
                 shield; it does NOT fire vs an immune spade enemy —
                 GameState.enemy_immunity, engine.py suit-power rule).
"""
from __future__ import annotations

import random
from typing import Sequence

import numpy as np

from ml.envs.action_mask import YIELD_ID, decode_action
from ml.envs.regicide_env import RegicideGymEnv


class BaselineAgent:
    """Base class: ``act(env, obs, info) -> int`` choosing only masked-in ids."""

    name = "baseline"

    def act(self, env: RegicideGymEnv, obs: np.ndarray, info: dict) -> int:
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Shared mask/game-state helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _legal_ids(info: dict) -> np.ndarray:
        mask = np.asarray(info["mask"], dtype=bool)
        legal = np.flatnonzero(mask)
        if legal.size == 0:
            raise RuntimeError(
                "BaselineAgent: empty legality mask — no legal action exists "
                "(R1/R2 carve-outs should prevent this; env bug if reached)"
            )
        return legal

    @staticmethod
    def _state(env: RegicideGymEnv):
        """(phase, hand, needed) of the pending decision from the engine state."""
        game_state = env.engine.game_state
        phase = env.engine.phase
        hand = game_state.get_active_player().hand
        needed = (
            game_state.current_enemy.effective_attack() if phase == "defense" else None
        )
        return phase, hand, needed

    @staticmethod
    def _decoded_cards(
        action_id: int, hand: Sequence, phase: str, needed: int | None
    ) -> tuple:
        return tuple(decode_action(int(action_id), hand, phase, needed)["cards"])

    def _choose_tie(self, candidates: list[int]) -> int:
        # Global random (run-seeded) — deterministic under seed_everything.
        return int(random.choice(candidates))


class RandomAgent(BaselineAgent):
    """Uniform choice over unmasked action ids."""

    name = "random"

    def act(self, env: RegicideGymEnv, obs: np.ndarray, info: dict) -> int:
        legal = self._legal_ids(info)
        return int(np.random.choice(legal))


class GreedyHighestAgent(BaselineAgent):
    """Max-value legal attack; minimal defense subset meeting ``needed``."""

    name = "greedy-highest"

    def act(self, env: RegicideGymEnv, obs: np.ndarray, info: dict) -> int:
        legal = self._legal_ids(info)
        phase, hand, needed = self._state(env)
        if phase == "defense":
            return self._best_defense(legal, hand, needed)
        return self._best_attack(legal, hand, env.engine.game_state)

    def _best_attack(
        self, legal: np.ndarray, hand: Sequence, game_state=None
    ) -> int:
        plays = [int(a) for a in legal if int(a) != YIELD_ID]
        if not plays:
            # R2 stuck carve-out: yield is the ONLY unmasked action.
            return YIELD_ID
        scored = [(self._attack_value(a, hand), a) for a in plays]
        best = max(value for value, _ in scored)
        return self._choose_tie([a for value, a in scored if value == best])

    def _attack_value(self, action_id: int, hand: Sequence) -> float:
        cards = self._decoded_cards(action_id, hand, "attack", None)
        return float(sum(card.value for card in cards))

    def _best_defense(self, legal: np.ndarray, hand: Sequence, needed: int) -> int:
        scored = [(self._defense_key(a, hand, needed), int(a)) for a in legal]
        best = min(key for key, _ in scored)
        return self._choose_tie([a for key, a in scored if key == best])

    def _defense_key(self, action_id: int, hand: Sequence, needed: int) -> tuple:
        cards = self._decoded_cards(action_id, hand, "defense", needed)
        total = sum(card.value for card in cards)
        # min total >= needed (minimal overspend); tie -> fewest cards.
        return (total, len(cards))


class ShieldGreedyAgent(GreedyHighestAgent):
    """GreedyHighest + spade preference while the enemy is unshielded."""

    name = "shield-greedy"

    def _best_attack(
        self, legal: np.ndarray, hand: Sequence, game_state=None
    ) -> int:
        plays = [int(a) for a in legal if int(a) != YIELD_ID]
        if not plays:
            return YIELD_ID
        if game_state is not None:
            plays = self._prefer_spades(plays, hand, game_state)
        scored = [(self._attack_value(a, hand), a) for a in plays]
        best = max(value for value, _ in scored)
        return self._choose_tie([a for value, a in scored if value == best])

    def _prefer_spades(
        self, plays: list[int], hand: Sequence, game_state
    ) -> list[int]:
        """Restrict to spade-containing plays while a spade would fire.

        A spade raises the enemy shield — valuable only while the shield is 0
        and the suit power actually fires (it does NOT fire vs an immune spade
        enemy: engine resolves spades only when
        ``enemy.suit != 'spades' or not GameState.enemy_immunity``).
        """
        enemy = game_state.current_enemy
        shield_down = enemy.shield == 0
        spade_fires = enemy.suit != "spades" or not game_state.enemy_immunity
        if not (shield_down and spade_fires):
            return plays
        spade_plays = [
            a
            for a in plays
            if any(card.suit == "spades" for card in self._decoded_cards(
                a, hand, "attack", None
            ))
        ]
        return spade_plays if spade_plays else plays


__all__ = ["BaselineAgent", "RandomAgent", "GreedyHighestAgent", "ShieldGreedyAgent"]
