"""Per-step and per-episode telemetry callbacks (Phase 3.2).

Persists the Phase 3.1 per-step info telemetry emitted by RegicideGymEnv:

- ``PerStepJSONLCallback``   -> ``steps.jsonl``: one JSON line per env step
  (timestep, env, episode, phase, action_idx, decoded_cards, damage_dealt,
  shield_raised, suit_powers, yield_taken, defense fields, reward).
- ``EpisodeScalarsCallback`` -> ``episodes.jsonl``: one JSON line per
  finished episode (win/loss, length, enemies_defeated, damage_per_turn,
  yield_rate, discard_overspend_mean, campaign_length); the same scalars
  are recorded to the SB3 logger (TensorBoard).

Both extend ``stable_baselines3.common.callbacks.BaseCallback`` and read the
vec-env rollout locals (``infos``/``rewards``/``dones``). SB3's DummyVecEnv
reports ``dones = terminated or truncated`` per env and sets
``info["TimeLimit.truncated"]``; the env's telemetry keys pass through the
ActionMasker/InvalidActionGuard/Monitor wrappers unchanged (Monitor only
adds ``info["episode"]`` on done). Infos without the ``phase`` telemetry key
(reset / invalid-action paths) are skipped.

``timestep`` is the callback's ``num_timesteps`` snapshot (the global env
timestep count after the vec-env batch stepped) — monotonic, shared by the
lines of one vec-env batch.
"""
from __future__ import annotations

import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback


def _ensure_open(file, path: Path):
    """Lazily open the JSONL sink (truncating)."""
    if file is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8")
    return file


class PerStepJSONLCallback(BaseCallback):
    """Write one JSONL line per env step from the rollout infos."""

    def __init__(self, path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.path = Path(path)
        self._file = None
        self._episode_ids: dict = {}  # env index -> current episode id (0-based)

    def _on_step(self) -> bool:
        self._file = _ensure_open(self._file, self.path)
        infos = self.locals.get("infos") or []
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        for env_idx, info in enumerate(infos):
            if "phase" not in info:
                continue  # non-telemetry info (reset / invalid-action path)
            record = {
                "timestep": int(self.num_timesteps),
                "env": env_idx,
                "episode": self._episode_ids.get(env_idx, 0),
                "phase": info["phase"],
                "action_idx": info["action_idx"],
                "decoded_cards": list(info["decoded_cards"]),
                "damage_dealt": info["damage_dealt"],
                "shield_raised": info["shield_raised"],
                "suit_powers": dict(info["suit_powers"]),
                "yield_taken": info["yield_taken"],
                "defense_needed": info["defense_needed"],
                "defense_discarded_sum": info["defense_discarded_sum"],
                "defense_overspend": info["defense_overspend"],
                "reward": float(rewards[env_idx]) if rewards is not None else None,
            }
            self._file.write(json.dumps(record) + "\n")
            if dones is not None and dones[env_idx]:
                self._episode_ids[env_idx] = self._episode_ids.get(env_idx, 0) + 1
                self._file.flush()
        return True

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class EpisodeScalarsCallback(BaseCallback):
    """Aggregate per-episode scalars at episode end -> episodes.jsonl + TB.

    The per-env accumulator buffer is created on the env's first telemetry
    step and finalized when that env reports a done (terminated OR
    truncated — a truncated episode is recorded with result "truncated",
    never as a win).
    """

    def __init__(self, path, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.path = Path(path)
        self._file = None
        self._buffers: dict = {}  # env index -> accumulator dict

    def _on_step(self) -> bool:
        self._file = _ensure_open(self._file, self.path)
        infos = self.locals.get("infos") or []
        dones = self.locals.get("dones")
        for env_idx, info in enumerate(infos):
            if "phase" not in info:
                continue
            buf = self._buffers.setdefault(
                env_idx, {"length": 0, "damage_sum": 0, "yields": 0, "overspends": []}
            )
            buf["length"] += 1
            buf["damage_sum"] += info["damage_dealt"]
            buf["yields"] += int(info["yield_taken"])
            if info["defense_overspend"] is not None:
                buf["overspends"].append(info["defense_overspend"])
            if dones is not None and dones[env_idx]:
                self._finish_episode(env_idx, info)
        return True

    def _finish_episode(self, env_idx: int, info: dict) -> None:
        buf = self._buffers.pop(env_idx)
        truncated = bool(info.get("TimeLimit.truncated", False))
        result = info.get("result")
        win = result == "win" and not truncated
        loss = result == "player_defeated" and not truncated
        overspends = buf["overspends"]
        overspend_mean = (
            sum(overspends) / len(overspends) if overspends else 0.0
        )
        record = {
            "timestep": int(self.num_timesteps),
            "env": env_idx,
            "win": win,
            "loss": loss,
            "result": "win" if win else ("truncated" if truncated else "player_defeated"),
            "length": buf["length"],
            "enemies_defeated": info["enemies_defeated_so_far"],
            "damage_per_turn": buf["damage_sum"] / buf["length"],
            "yield_rate": buf["yields"] / buf["length"],
            "discard_overspend_mean": overspend_mean,
            "campaign_length": info.get("campaign_length"),
        }
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()
        self._record_scalars(record)

    def _record_scalars(self, record: dict) -> None:
        self.logger.record("telemetry/win", float(record["win"]))
        self.logger.record("telemetry/episode_length", record["length"])
        self.logger.record("telemetry/enemies_defeated", record["enemies_defeated"])
        self.logger.record("telemetry/damage_per_turn", record["damage_per_turn"])
        self.logger.record("telemetry/yield_rate", record["yield_rate"])
        self.logger.record(
            "telemetry/discard_overspend_mean", record["discard_overspend_mean"]
        )
        if record["campaign_length"] is not None:
            self.logger.record("telemetry/campaign_length", record["campaign_length"])

    def _on_training_end(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


__all__ = ["PerStepJSONLCallback", "EpisodeScalarsCallback"]
