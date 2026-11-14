"""Regicide Gymnasium environment (Phase 1, Step 1.5).

Wraps the existing Regicide Engine (src/game/engine.py) in a pure Gymnasium
env. The env recomputes observations (obs_encoder) and rewards (PBRS via
reward.py) env-side; the engine's own reward values NEVER pass through. The
engine's stdout noise (cli.display prints) is silenced per call with
redirect_stdout, and the engine logger is a no-op (M1) so no regicide.log
spam occurs during training.

Contract
--------
    RegicideGymEnv(EnvConfig)   gymnasium.Env
    EnvConfig(num_players=2, campaign_length=3, reward_config_name=...,
              gamma=0.99, seed=None, step_limit=1000, jester_mode="random")

- observation_space: Box(0, 1, (108,), float32)  (obs_encoder.OBS_DIM)
- action_space:     Discrete(172)                (action_mask.N_ACTIONS)
- NO Engine is built in __init__ — the Engine is built on reset so that
  seeds and campaign config changes apply to every episode.

Campaign truncation
-------------------
``campaign_length`` L slices the castle deck to ``L - 1`` remaining cards
after Engine.reset (the first enemy is already drawn at GameState init), so
the campaign contains exactly L enemies: 1 in play + L-1 in the castle.

Determinism (M5, run-level)
---------------------------
``reset(seed=s)`` seeds the GLOBAL ``random`` module once per reset when a
seed is provided — the engine's deck shuffles all read global random, so one
seed governs the whole episode. ``reset(seed=None)`` continues the RNG
stream (no re-seed). The Jester next-player chooser is a private
``random.Random(seed)`` instance (jester_mode="random"), deterministic per
episode seed. NOTE: because the RNG state is global, episodes are
deterministic only when run in the same call sequence (run-level, not
instance-level, determinism) — eval harnesses reset(seed=s) per episode and
act with deterministic=True (Phase 2.3 gate).

Review amendments implemented here
----------------------------------
- R1 (defense auto-loss): when a defense phase is entered (or stepped into)
  with an empty hand or Σhand < effective_attack, the env performs the
  auto-loss ITSELF via engine.step({"type": "defend", "cards": []}) — the
  (potentially all-False) defense mask is never presented. Chained into the
  same env step: the returned transition goes straight from the pre-step
  attack state to the terminal loss, skipping the intermediate defense state
  for Φ. Mask-equivalence carve-out documented.
- R2 (stuck-state mask): when building the attack mask the env passes
  ``stuck=GameState.player_is_stuck(active)`` — build_mask then unmasks
  YIELD ONLY (index 0). Engine.step preempts a stuck turn with the loss
  before parsing the action, so the decoded yield resolves the loss. This
  unmasked yield does NOT pass can_yield semantics — a documented exception
  for the Phase 1.6 mask-equivalence gate. In non-stuck attack states the
  env also passes ``yield_allowed=GameState.can_yield(active)`` so the
  engine-refused yield ({"error": "Yield forbidden"}) is masked out.
- M1 (no-op logger): the env constructs the Engine with a disabled
  logging.Logger (NullHandler, propagation off) and a get_logger returning
  the same — the engine logs full deck dumps per step and must not spam
  regicide.log or stdout during training.

Exact-defeat detection
----------------------
The engine's step info carries only ``result`` — no exactness flag. Exact
defeat is detected deck-side (read-only): engine.handle_enemy_defeat pushes
the defeated enemy card onto the tavern TOP (add_to_top → index 0) on an
exact defeat and into the discard pile otherwise. The env captures the
pre-step ``current_enemy`` identity and, if the result is
"enemy_defeated" or "win" (the final enemy also passes through
handle_enemy_defeat before the win), declares exact_defeat iff that exact
Card object is now the tavern top. Identity (``is``) is used, not value
equality, so distinct same-rank cards cannot collide.

Per-step telemetry (Phase 3.1)
------------------------------
Every step's info additionally carries READ-ONLY telemetry keys derived from
the engine state (the engine is never mutated for telemetry):

- ``phase``: the pre-step phase the action was taken in.
- ``action_idx``: the action index passed to step() (for the R1 auto-loss
  the executed action is the env's empty defend, not this index).
- ``decoded_cards``: display names of the cards the decoded action plays /
  discards (empty for yield and the R1 auto-loss).
- ``yield_taken``: True iff this step was an attack-phase yield.
- ``damage_dealt``: pre/post ``current_damage`` delta; on defeat the PRE-step
  hp remaining (the engine replaces the enemy and the old object's
  current_damage carries the overshoot — hp drops to exactly 0, so hp
  remaining is the honest hp drop).
- ``shield_raised``: hybrid derivation, documented for robustness — when the
  enemy persists, the pre/post ``enemy.shield`` delta (exact, and also covers
  the jester's retroactive pending-spade resolution, which decoded cards
  cannot see); when the enemy is replaced by defeat, card-derived (the
  engine scopes a firing spade's shield to the WHOLE play's value —
  handle_suit_powers passes the play's total value to resolve_suit_power).
  A spade blocked by immunity is pending, not raised.
- ``suit_powers``: dict suit -> "fires"/"blocked"/"pending" for the suits in
  this play, mirroring handle_suit_powers + the pre-jester pending-spade
  collection: fires when the enemy is not immune (is_enemy_immune with the
  pre-step face_jester_played — a jester never shares a play with other
  cards); a blocked SPADE is "pending" (collected for retroactive jester
  resolution); blocked hearts/diamonds are "blocked". Clubs have no suit
  power (damage doubling only) and are omitted.
- ``jester_event``: {"played": bool, "next_player": name of the chosen next
  player (the post-step active player) or None}.
- ``defense_needed`` / ``defense_discarded_sum`` / ``defense_overspend``:
  the defense phase's pre-step effective_attack, the value-sum of the
  discarded cards, and their difference (negative = shortfall, i.e. the R1
  empty-defend auto-loss records 0 discarded). None on attack-phase steps.
- ``defeat_cause``: on terminal steps only — "exact_win"/"win",
  "uncoverable_defense" (R1 env auto-loss), "insufficient_defense"
  (engine defense_validation loss outside the R1 path; effectively
  unreachable through the masked env), "stuck" (engine preempts a
  stuck turn), "truncated" (env step_limit). None on non-terminal steps.
- ``step_count`` / ``enemies_defeated_so_far``: 1-based episode step number
  and the enemies defeated so far this episode (env-side counters).
"""
from __future__ import annotations

import contextlib
import io
import logging
import random
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from game.engine import Engine
from game.game_state import GameState
from game.player import Player
from game.rules import is_enemy_immune

from ml.envs import obs_encoder, reward
from ml.envs.action_mask import N_ACTIONS, build_mask, decode_action
from ml.envs.obs_encoder import OBS_DIM
from ml.envs.reward import (
    AGGRESSIVE_SHAPING,
    MINIMAL_SHAPING,
    SPARSE_CURRICULUM,
)

HAND_LIMITS = {2: 7, 3: 6, 4: 5}
DEFAULT_STEP_LIMIT = 1000
NOOP_LOGGER_NAME = "regicide.env.silent"

REWARD_CONFIGS = {
    config.name: config
    for config in (AGGRESSIVE_SHAPING, MINIMAL_SHAPING, SPARSE_CURRICULUM)
}


@dataclass
class EnvConfig:
    """Environment configuration (Step 1.5 contract).

    gamma MUST equal the training algorithm's discount factor (PBRS
    requirement — see ml/envs/reward.py). reward_config_name indexes the
    named RewardConfigs defined in reward.py.
    """

    num_players: int = 2
    campaign_length: int = 3
    reward_config_name: str = "aggressive_shaping"
    gamma: float = 0.99
    seed: Optional[int] = None
    step_limit: int = DEFAULT_STEP_LIMIT
    jester_mode: str = "random"


class _SilentEngineLogger(logging.Logger):
    """M1: logging.Logger subclass matching the engine's custom logger
    interface (info/warning/error/data — the engine logs full deck dumps per
    step). Every method is a no-op, so nothing reaches regicide.log or
    stdout during training."""

    def info(self, *args, **kwargs) -> None:
        pass

    def warning(self, *args, **kwargs) -> None:
        pass

    def error(self, *args, **kwargs) -> None:
        pass

    def data(self, *args, **kwargs) -> None:
        pass


def _make_engine_logger() -> _SilentEngineLogger:
    logger = _SilentEngineLogger(NOOP_LOGGER_NAME)
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger


def _noop_get_logger(name: Optional[str] = None) -> "_SilentEngineLogger":
    return _make_engine_logger()


class RegicideGymEnv(gym.Env):
    """Gymnasium env over the Regicide Engine (Step 1.5)."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else EnvConfig()
        cfg = self.config
        # Boundary validation of the externally supplied config.
        if cfg.num_players not in HAND_LIMITS:
            raise ValueError(
                f"num_players must be one of {sorted(HAND_LIMITS)}, got {cfg.num_players}"
            )
        if not isinstance(cfg.campaign_length, int) or cfg.campaign_length < 1:
            raise ValueError(
                f"campaign_length must be a positive int, got {cfg.campaign_length!r}"
            )
        if cfg.reward_config_name not in REWARD_CONFIGS:
            raise ValueError(
                f"reward_config_name must be one of {sorted(REWARD_CONFIGS)}, "
                f"got {cfg.reward_config_name!r}"
            )
        if not 0.0 <= cfg.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {cfg.gamma}")
        if cfg.jester_mode != "random":
            raise ValueError(
                f"jester_mode must be 'random' in Phase 1, got {cfg.jester_mode!r}"
            )
        if not isinstance(cfg.step_limit, int) or cfg.step_limit < 1:
            raise ValueError(f"step_limit must be a positive int, got {cfg.step_limit!r}")

        self.reward_config = REWARD_CONFIGS[cfg.reward_config_name]
        self.observation_space = spaces.Box(0, 1, (OBS_DIM,), np.float32)
        self.action_space = spaces.Discrete(N_ACTIONS)
        # No Engine here — built on reset so seeds/campaign changes apply.
        self.engine: Optional[Engine] = None
        self._prev_potential: Optional[float] = None
        self._step_count = 0
        self._enemies_defeated = 0

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset to a fresh episode. super().reset(seed=seed) follows the
        Gymnasium convention; a provided seed additionally seeds the global
        random module ONCE (M5 run-level determinism — see module docstring).
        """
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)
        self.engine = self._build_engine(seed)
        with contextlib.redirect_stdout(io.StringIO()):
            players = self._build_players()
            self.engine.reset(players)
            # Campaign truncation: the first enemy is already drawn at
            # GameState init, so keep L-1 castle cards -> exactly L enemies.
            game_state = self.engine.game_state
            game_state.castle_deck.cards = game_state.castle_deck.cards[
                : self.config.campaign_length - 1
            ]
        self._step_count = 0
        self._enemies_defeated = 0
        self._prev_potential = reward.potential(game_state, self.reward_config)
        obs = self._current_obs()
        mask = self.current_mask()
        info = {
            "mask": mask,
            "action_mask": mask,
            "campaign_length": self.config.campaign_length,
        }
        return obs, info

    def step(self, action_id: int):
        """One env step. Returns the 5-tuple (obs, reward, terminated,
        truncated, info) with PBRS reward (engine reward ignored)."""
        if self.engine is None:
            raise RuntimeError("reset() must be called before step()")
        if self.engine.done:
            raise RuntimeError("Episode is done — call reset() before stepping again")
        if isinstance(action_id, np.integer):
            action_id = int(action_id)
        if not isinstance(action_id, int) or not 0 <= action_id < N_ACTIONS:
            # Out-of-range action at the boundary — treat as invalid, not a crash.
            return self._invalid_action_response()

        game_state = self.engine.game_state
        phase = self.engine.phase
        active = game_state.get_active_player()

        # R1 (step-entry guard): an unplayable defense state is auto-lost
        # directly — the (possibly all-False) defense mask is never used.
        if phase == "defense" and _defense_insufficient(game_state):
            return self._step_engine(
                {"type": "defend", "cards": []}, action_id, r1_auto_loss=True
            )

        needed = (
            game_state.current_enemy.effective_attack() if phase == "defense" else None
        )
        stuck = (
            game_state.player_is_stuck(active) if phase == "attack" else False
        )
        mask = build_mask(
            phase,
            active.hand,
            needed=needed,
            stuck=stuck,
            yield_allowed=game_state.can_yield(active),
        )
        if not mask[action_id]:
            # Masked action selected — unreachable with ActionMasker.
            return self._invalid_action_response()

        action_dict = decode_action(action_id, active.hand, phase, needed)
        return self._step_engine(action_dict, action_id)

    def action_masks(self) -> np.ndarray:
        """Mask for the CURRENT pending state (SB3 ActionMasker interface)."""
        return self.current_mask()

    def render(self):  # minimal — Phase 1 has no rendering
        return None

    def close(self) -> None:  # nothing to release
        pass

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _build_engine(self, seed: Optional[int]) -> Engine:
        """Engine with a no-op logger (M1) and an injected seeded Jester
        chooser (jester_mode="random" — picks a random player from the list)."""
        chooser_rng = random.Random(seed)

        def chooser(players: list) -> Player:
            return chooser_rng.choice(players)

        return Engine(
            _make_engine_logger(),
            _noop_get_logger,
            next_player_chooser=chooser,
        )

    def _build_players(self) -> list:
        hand_limit = HAND_LIMITS[self.config.num_players]
        return [
            Player(name=f"Player{i + 1}", hand_limit=hand_limit)
            for i in range(self.config.num_players)
        ]

    def _current_obs(self) -> np.ndarray:
        return obs_encoder.encode(self.engine.game_state, self.engine.phase)

    def current_mask(self) -> np.ndarray:
        """Legality mask for the CURRENT pending state (post-step obs).

        Returns an all-False mask once the episode is over (no legal actions
        remain in a terminal state).
        """
        if self.engine is None or self.engine.done:
            return np.zeros(N_ACTIONS, dtype=bool)
        game_state = self.engine.game_state
        phase = self.engine.phase
        active = game_state.get_active_player()
        needed = (
            game_state.current_enemy.effective_attack() if phase == "defense" else None
        )
        stuck = game_state.player_is_stuck(active) if phase == "attack" else False
        return build_mask(
            phase,
            active.hand,
            needed=needed,
            stuck=stuck,
            yield_allowed=game_state.can_yield(active),
        )

    def _invalid_action_response(self):
        """Masked/out-of-range action selected — should be unreachable with
        ActionMasker. Log a warning and return an invalid-step response."""
        logging.getLogger(__name__).warning("Invalid action selected")
        return self._current_obs(), -1.0, False, False, {"error": "Invalid action"}

    def _step_engine(self, action_dict: dict, action_id: int, r1_auto_loss: bool = False):
        """Run the engine transition (stdout silenced), compute PBRS reward
        and events, build the next obs/mask, handle R1 chaining, and attach
        the per-step telemetry keys (Phase 3.1) to info."""
        pre = _capture_pre_step(self.engine)
        with contextlib.redirect_stdout(io.StringIO()):
            _, _, done, engine_info = self.engine.step(action_dict)

        game_state = self.engine.game_state
        result = engine_info.get("result")

        # R1 (post-step chain): an attack step that just entered the defense
        # phase with an unplayable hand auto-loses via the empty defend —
        # chained into THIS env step so the insufficient-defense mask is
        # never presented.
        if not done and self.engine.phase == "defense" and _defense_insufficient(
            game_state
        ):
            with contextlib.redirect_stdout(io.StringIO()):
                _, _, done, engine_info = self.engine.step(
                    {"type": "defend", "cards": []}
                )
            result = engine_info.get("result")
            r1_auto_loss = True

        events = _build_events(result, pre["enemy"], game_state)

        self._step_count += 1
        enemy_defeated = pre["enemy"].current_damage >= pre["enemy"].max_health
        if enemy_defeated:
            self._enemies_defeated += 1
        truncated = self._step_count >= self.config.step_limit and not done

        reward_value = reward.compute_reward(
            game_state,
            game_state,
            self._prev_potential,
            self.reward_config,
            self.config.gamma,
            events,
        )
        # Φ(s′) cache for the next transition. At terminal this value is
        # irrelevant — reset() re-caches Φ(s₀).
        self._prev_potential = reward.potential(game_state, self.reward_config)

        obs = self._current_obs()
        mask = self.current_mask()
        info = {
            "mask": mask,
            "action_mask": mask,
            "result": result,
            "campaign_length": self.config.campaign_length,
        }
        info.update(
            _build_telemetry(
                pre=pre,
                action_dict=action_dict,
                action_id=action_id,
                game_state=game_state,
                result=result,
                done=bool(done),
                truncated=bool(truncated),
                r1_auto_loss=r1_auto_loss,
                events=events,
                step_count=self._step_count,
                enemies_defeated=self._enemies_defeated,
                enemy_defeated=enemy_defeated,
            )
        )
        return obs, reward_value, bool(done), bool(truncated), info


def _defense_insufficient(game_state: GameState) -> bool:
    """R1 predicate: empty hand or Σhand < the enemy's effective attack."""
    active = game_state.get_active_player()
    needed = game_state.current_enemy.effective_attack()
    if active.hand_size() == 0:
        return True
    return sum(card.value for card in active.hand) < needed


def _build_events(
    result: Optional[str],
    prev_enemy: Optional[object],
    game_state: GameState,
) -> dict:
    """Events dict for reward.compute_reward.

    exact_defeat: deck-side identity check — an exact defeat pushes the
    defeated enemy Card onto the tavern TOP (engine.handle_enemy_defeat);
    a non-exact defeat sends it to the discard pile. Applies to
    "enemy_defeated" and to "win" (the final enemy also passes through
    handle_enemy_defeat before the win).
    """
    events: dict = {}
    if result in ("enemy_defeated", "win") and prev_enemy is not None:
        tavern_top = game_state.tavern_deck.cards[:1]
        if tavern_top and tavern_top[0] is prev_enemy:
            events["exact_defeat"] = True
    if result == "win":
        events["terminal"] = "win"
    elif result == "player_defeated":
        events["terminal"] = "loss"
    return events


# --------------------------------------------------------------------- #
# Per-step telemetry (Phase 3.1) — all values derived READ-ONLY from the
# engine state; the engine is never mutated for telemetry purposes.
# --------------------------------------------------------------------- #

DEFEAT_CAUSES = (
    "exact_win",
    "win",
    "uncoverable_defense",
    "insufficient_defense",
    "stuck",
    "truncated",
)


def _capture_pre_step(engine: Engine) -> dict:
    """Snapshot the pre-step state needed for telemetry derivation.

    Captured BEFORE engine.step mutates anything: the current enemy object
    (its damage/shield fields persist through replacement, which is what
    makes the defeat-time derivations below robust), pre-step phase, active
    player name, stuck flag, defense needed value, and the jester-immunity
    gate (face_jester_played) as it stood for this play.
    """
    game_state = engine.game_state
    phase = engine.phase
    enemy = game_state.current_enemy
    active = game_state.get_active_player()
    return {
        "enemy": enemy,
        "damage": enemy.current_damage,
        "shield": enemy.shield,
        "face_jester": game_state.face_jester_played,
        "phase": phase,
        "active_name": active.name,
        "stuck": (
            game_state.player_is_stuck(active) if phase == "attack" else False
        ),
        "needed": enemy.effective_attack() if phase == "defense" else None,
    }


def _derive_suit_powers(cards: list, pre: dict) -> dict:
    """Map suit -> "fires"/"blocked"/"pending" for the suits in this play.

    Mirrors engine.handle_suit_powers plus the pre-jester pending-spade
    collection (engine._execute_play): a suit power fires when the enemy is
    not immune to the card's suit (is_enemy_immune with the PRE-step
    face_jester_played — a jester can never share a play with other cards);
    a blocked SPADE is collected into pending_spades for retroactive
    resolution by a later jester ("pending"); blocked hearts/diamonds simply
    do nothing ("blocked"). Clubs have no suit power (they only double
    damage via rules.total_attack) and are omitted.
    """
    powers: dict = {}
    for suit in ("hearts", "diamonds", "spades"):
        suit_cards = [c for c in cards if c.suit == suit]
        if not suit_cards:
            continue
        if is_enemy_immune(
            pre["enemy"], suit_cards[0], jester_active=pre["face_jester"]
        ):
            powers[suit] = "pending" if suit == "spades" else "blocked"
        else:
            powers[suit] = "fires"
    return powers


def _card_derived_shield(cards: list, pre: dict) -> int:
    """Card-derived shield value for the step where the enemy was replaced
    by defeat (the pre/post shield delta is unavailable across the
    replacement — the new enemy starts at shield 0).

    The engine scopes the spade shield to the WHOLE play's value
    (engine.handle_suit_powers passes the play's total value to
    rules.resolve_suit_power, which calls enemy.apply_shield(value)), so a
    firing spade shields the sum of ALL played card values. A spade blocked
    by immunity is collected as pending — no shield this play.
    """
    spades = [c for c in cards if c.suit == "spades"]
    if not spades:
        return 0
    if is_enemy_immune(
        pre["enemy"], spades[0], jester_active=pre["face_jester"]
    ):
        return 0
    return sum(c.value for c in cards)


def _derive_shield_raised(cards: list, pre: dict, game_state: GameState) -> int:
    """shield_raised derivation (robust hybrid, documented):

    - Enemy persists (not replaced): post-minus-pre ``enemy.shield`` delta.
      This is exact and ALSO covers the jester's retroactive pending-spade
      resolution (engine.handle_jester_played), which the decoded cards
      alone cannot see (a jester play contains no spades).
    - Enemy replaced by defeat: card-derived value (_card_derived_shield) —
      the delta is meaningless across the replacement, while a spade firing
      in the defeating play always shields before handle_enemy_defeat.
    """
    if game_state.current_enemy is pre["enemy"]:
        return pre["enemy"].shield - pre["shield"]
    return _card_derived_shield(cards, pre)


def _derive_damage_dealt(pre: dict, enemy_defeated: bool) -> int:
    """Damage dealt this step.

    Normal steps: pre/post ``current_damage`` delta (nothing but damage
    ever changes current_damage). Defeat steps: the PRE-step hp remaining —
    the engine replaces the enemy on defeat (current_damage of the old
    object carries the overshoot), and hp drops to exactly 0, so the
    hp remaining is the honest hp drop for the step.
    """
    if enemy_defeated:
        return pre["enemy"].max_health - pre["damage"]
    return pre["enemy"].current_damage - pre["damage"]


def _defeat_cause(
    result: Optional[str],
    done: bool,
    truncated: bool,
    events: dict,
    pre: dict,
    r1_auto_loss: bool,
) -> Optional[str]:
    """Terminal-step defeat cause; None on non-terminal steps.

    Mapping:
    - truncated            -> "truncated" (env step_limit, not an engine done)
    - result "win"         -> "exact_win" iff the exact-defeat deck-side
                              identity check fired (defeated enemy card is
                              the tavern top), else "win"
    - attack-phase loss    -> "stuck" (engine._begin_turn preempted the turn:
                              empty hand AND yield forbidden)
    - R1 env auto-loss     -> "uncoverable_defense" (empty hand or Σhand <
                              needed at defense entry — the env performed
                              the empty-defend loss itself)
    - any other loss       -> "insufficient_defense" (engine defense_validation
                              loss outside the R1 path; effectively
                              unreachable through the masked env because
                              uncovered defense subsets are masked out —
                              kept for direct-engine parity)
    """
    if truncated:
        return "truncated"
    if not done:
        return None
    if result == "win":
        return "exact_win" if events.get("exact_defeat") else "win"
    if pre["stuck"]:
        return "stuck"
    if r1_auto_loss:
        return "uncoverable_defense"
    return "insufficient_defense"


def _build_telemetry(
    pre: dict,
    action_dict: dict,
    action_id: int,
    game_state: GameState,
    result: Optional[str],
    done: bool,
    truncated: bool,
    r1_auto_loss: bool,
    events: dict,
    step_count: int,
    enemies_defeated: int,
    enemy_defeated: bool,
) -> dict:
    """Assemble the Phase 3.1 per-step telemetry info keys (read-only)."""
    cards = action_dict.get("cards", [])
    decoded_cards = [c.display_name for c in cards]
    is_yield = (
        pre["phase"] == "attack"
        and action_dict.get("type") == "play"
        and not cards
    )
    is_jester = len(cards) == 1 and cards[0].c_type == "jester"

    if pre["phase"] == "defense":
        discarded_sum = sum(c.value for c in cards)
        defense_needed: Optional[int] = pre["needed"]
        defense_discarded_sum: Optional[int] = discarded_sum
        # Negative overspend = shortfall (the R1 empty-defend auto-loss
        # records 0 discarded against the needed value).
        defense_overspend: Optional[int] = discarded_sum - pre["needed"]
    else:
        defense_needed = None
        defense_discarded_sum = None
        defense_overspend = None

    return {
        "phase": pre["phase"],
        "action_idx": action_id,
        "decoded_cards": decoded_cards,
        "yield_taken": is_yield,
        "damage_dealt": _derive_damage_dealt(pre, enemy_defeated),
        "shield_raised": _derive_shield_raised(cards, pre, game_state),
        "suit_powers": _derive_suit_powers(cards, pre),
        "jester_event": {
            "played": is_jester,
            "next_player": (
                game_state.get_active_player().name if is_jester else None
            ),
        },
        "defense_needed": defense_needed,
        "defense_discarded_sum": defense_discarded_sum,
        "defense_overspend": defense_overspend,
        "defeat_cause": _defeat_cause(
            result, done, truncated, events, pre, r1_auto_loss
        ),
        "step_count": step_count,
        "enemies_defeated_so_far": enemies_defeated,
    }
