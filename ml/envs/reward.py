"""Potential-based reward shaping (PBRS) for the Regicide Gym env.

This module is env-side recomputation ONLY: it operates on GameState
snapshots and never touches the engine. The engine's own reward values
NEVER pass through — the −0.01 engine step penalty is absorbed by ignoring
engine rewards entirely.

Φ(s) (potential), parameterized by RewardConfig weights w1..w5:

    Φ(s) = w1·(enemies_defeated / 12)
         + w2·(1 − enemy_hp_fraction)
         + w3·(Σ team hand values / (players × hand_limit × 20))
         + w4·(tavern_size / 40)
         − w5·(discard_size / 40)

Term notes:
- enemies_defeated is NOT stored on GameState; it is derived exactly like
  the web bridge does: 12 − len(castle_deck.cards) − (1 if enemy in play).
  Caveat: env-side campaign truncation (castle sliced to L enemies) shifts
  the absolute value (Φ(s_0) reads 12−L instead of 0). PBRS policy
  invariance only depends on Φ differences within an episode, so the
  constant per-episode offset is documented and accepted.
- enemy_hp_fraction = current_damage / max_health of the enemy in play.
  current_enemy None ⇔ campaign won (GameState.is_game_won) → Φ returns
  0 by the terminal convention below (hp term 0 included).
- hand values normalize by 20 — the same ÷20 convention as the observation
  encoder (review amendment R3). Tavern hands hold values ≤ 10; a royal
  (value 15/20) would read 0.75 under the same denominator. The
  denominator uses the sum of each player's hand_limit, which equals
  players × hand_limit for the uniform limits the engine assigns.
- tavern/discard sizes normalize by 40 (the 2-player tavern size).

Reward (compute_reward):

    reward = γ·Φ(s′) − Φ(s)
           + [exact_defeat_bonus if events["exact_defeat"]]
           + [win_reward if events["terminal"] == "win"]
           + [loss_reward if events["terminal"] == "loss"]

Terminal interaction (PBRS convention, implemented exactly):
- Φ(s′) is forced to 0 at terminal: won states already return 0 from
  potential(); loss states are indistinguishable from mid-game states
  (GameState carries no loss marker) so events["terminal"] == "loss"
  forces Φ(s′) = 0 here.
- The shaped part therefore collapses to −Φ(s_prev) at terminal, and the
  terminal ±10 is ADDED ON TOP: reward = ±10 − Φ(s_prev) + bonuses.
- The env MUST pass events["terminal"] on terminal steps for the ±10 to
  apply; a won state without the event still shapes correctly (Φ(s′)=0
  via potential()) but receives no terminal bonus.

γ REQUIREMENT (prominent): gamma is passed by the env from the TRAINING
config and MUST equal the algorithm's discount factor (MaskablePPO gamma,
0.99 in the Phase 2 config). PBRS policy invariance (Ng et al. 1999) holds
only when the shaping discount equals the learning discount. Never
hardcode gamma here.

Truncation (review amendment M2): the step-limit truncation (1000 steps,
truncated=True) is treated as NON-terminal for Φ purposes — the env passes
no terminal event, the shaped delta keeps bootstrapping with Φ(s′), and no
±10 fires. The residual Φ bias at truncation is accepted as rare + small.

events contract (dict built by the env from engine step info):
- "exact_defeat": truthy → +exact_defeat_bonus outside Φ (engine
  rules.exact_defeat semantics: total damage exactly == max_health).
- "terminal": "win" or "loss" on the final step only, absent otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Optional

if TYPE_CHECKING:
    from game.game_state import GameState

# Full castle size at reset (J/Q/K × 4 suits) — matches the web bridge.
CASTLE_TOTAL = 12
# Card-value normalization denominator (÷20 convention, matches obs encoder).
VALUE_DENOMINATOR = 20
# Tavern/discard pile size normalization denominators (2-player tavern size).
TAVERN_DENOMINATOR = 40
DISCARD_DENOMINATOR = 40

__all__ = [
    "RewardConfig",
    "AGGRESSIVE_SHAPING",
    "MINIMAL_SHAPING",
    "SPARSE_CURRICULUM",
    "potential",
    "compute_reward",
]


@dataclass(frozen=True)
class RewardConfig:
    """Weights for Φ(s) and the fixed terminal/event bonus values."""

    name: str
    w1: float  # enemies_defeated fraction
    w2: float  # enemy hp depletion
    w3: float  # team hand value fraction
    w4: float  # tavern size
    w5: float  # discard size (penalty)
    exact_defeat_bonus: float = 1.0
    win_reward: float = 10.0
    loss_reward: float = -10.0


AGGRESSIVE_SHAPING = RewardConfig(
    name="aggressive_shaping", w1=2.0, w2=1.0, w3=0.3, w4=0.2, w5=0.1
)
MINIMAL_SHAPING = RewardConfig(
    name="minimal_shaping", w1=1.0, w2=0.5, w3=0.0, w4=0.0, w5=0.0
)
SPARSE_CURRICULUM = RewardConfig(
    name="sparse_curriculum", w1=0.0, w2=0.0, w3=0.0, w4=0.0, w5=0.0
)


def _enemies_defeated(gs: GameState) -> int:
    """Derive defeated-enemy count from the castle (bridge formula).

    GameState does not store this: 12 − castle_remaining − enemy_in_play.
    """
    in_play = 1 if gs.current_enemy is not None else 0
    return CASTLE_TOTAL - len(gs.castle_deck.cards) - in_play


def potential(gs: GameState, config: RewardConfig) -> float:
    """Φ(s) — pure function of a GameState snapshot. Reads only; mutates nothing."""
    if gs.is_game_won():
        # Φ(terminal) = 0 convention: current_enemy is None (campaign won),
        # so the hp term is 0 and the whole potential collapses to 0.
        return 0.0

    defeated_frac = _enemies_defeated(gs) / CASTLE_TOTAL
    enemy = gs.current_enemy
    hp_frac = 1.0 - enemy.current_damage / enemy.max_health
    hand_capacity = sum(player.hand_limit for player in gs.players)
    hand_value = sum(card.value for player in gs.players for card in player.hand)

    return (
        config.w1 * defeated_frac
        + config.w2 * hp_frac
        + config.w3 * (hand_value / (hand_capacity * VALUE_DENOMINATOR))
        + config.w4 * (len(gs.tavern_deck.cards) / TAVERN_DENOMINATOR)
        - config.w5 * (len(gs.discard_pile.cards) / DISCARD_DENOMINATOR)
    )


def compute_reward(
    prev_gs: GameState,
    next_gs: GameState,
    prev_potential: float,
    config: RewardConfig,
    gamma: float,
    events: Optional[Mapping[str, Any]] = None,
) -> float:
    """PBRS reward for one transition: γ·Φ(s′) − Φ(s) + bonuses.

    prev_gs is accepted for signature symmetry with the env's transition
    tuple; Φ(s) itself comes from the cached prev_potential the env computed
    on the previous step (Φ is never recomputed here).

    gamma MUST be the algorithm's own discount factor (see module docstring).
    """
    if not 0.0 <= gamma <= 1.0:
        raise ValueError(
            f"gamma must be in [0, 1] and equal the algorithm discount (PBRS), got {gamma}"
        )
    events = events or {}

    terminal = events.get("terminal")
    if terminal is not None and terminal not in ("win", "loss"):
        raise ValueError(f"events['terminal'] must be 'win' or 'loss', got {terminal!r}")

    # Φ(s′) = 0 at terminal (see module docstring for the exact convention).
    next_phi = 0.0 if terminal is not None else potential(next_gs, config)
    reward = gamma * next_phi - prev_potential

    if events.get("exact_defeat"):
        reward += config.exact_defeat_bonus
    if terminal == "win":
        reward += config.win_reward
    elif terminal == "loss":
        reward += config.loss_reward
    return reward
