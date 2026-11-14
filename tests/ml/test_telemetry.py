"""Tests for Phase 3 telemetry — Step 3.1 (per-step info keys in
RegicideGymEnv) and Step 3.2 (PerStepJSONLCallback + EpisodeScalarsCallback
persisted by the PPO trainer).

Env-side tests drive scripted episodes with mask-legal actions (mirroring
tests/ml/test_env.py helpers). Callback tests share one module-scoped
2k-step training run (the standard smoke budget) and parse its JSONL/TB
artifacts from a tmp dir.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from ml.envs.action_mask import decode_action
from ml.envs.regicide_env import EnvConfig, RegicideGymEnv
from ml.trainers.ppo_trainer import TrainingConfig, train

REQUIRED_PER_STEP_KEYS = {
    "phase",
    "action_idx",
    "decoded_cards",
    "yield_taken",
    "damage_dealt",
    "shield_raised",
    "suit_powers",
    "jester_event",
    "defense_needed",
    "defense_discarded_sum",
    "defense_overspend",
    "defeat_cause",
    "step_count",
    "enemies_defeated_so_far",
}

DEFEAT_CAUSES = {
    "exact_win",
    "win",
    "uncoverable_defense",
    "insufficient_defense",
    "stuck",
    "truncated",
}


def _collect_episode(env: RegicideGymEnv, seed: int, policy: str = "random"):
    """Step a fresh episode with mask-legal actions until terminal.

    Returns (infos, terminated, truncated) — one info dict per step.
    """
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    infos = []
    terminated = truncated = False
    for _ in range(env.config.step_limit):
        mask = info["mask"]
        legal = np.flatnonzero(mask)
        if policy == "zero_single":
            singles = legal[(legal >= 1) & (legal <= 7)]
            action = int(singles[0]) if len(singles) else int(legal[0])
        else:
            action = int(rng.choice(legal))
        obs, reward, terminated, truncated, info = env.step(action)
        infos.append(info)
        if terminated or truncated:
            break
    return infos, terminated, truncated


# --------------------------------------------------------------------- #
# Step 3.1 — env-side per-step info
# --------------------------------------------------------------------- #
def test_info_contains_required_per_step_fields():
    env = RegicideGymEnv(EnvConfig(seed=0))
    player_names = {f"Player{i}" for i in range(1, env.config.num_players + 1)}
    for seed in range(8):
        infos, terminated, truncated = _collect_episode(env, seed)
        assert terminated or truncated, f"episode seed={seed} never terminated"
        for i, info in enumerate(infos):
            assert REQUIRED_PER_STEP_KEYS <= set(info)
            assert info["phase"] in ("attack", "defense")
            assert isinstance(info["decoded_cards"], list)
            assert info["damage_dealt"] >= 0
            assert info["shield_raised"] >= 0
            assert info["yield_taken"] in (True, False)
            assert set(info["suit_powers"].values()) <= {"fires", "blocked", "pending"}
            assert info["jester_event"]["played"] in (True, False)
            if info["jester_event"]["played"]:
                assert info["jester_event"]["next_player"] in player_names
            else:
                assert info["jester_event"]["next_player"] is None
            assert info["step_count"] == i + 1
            assert info["enemies_defeated_so_far"] >= 0
            if i < len(infos) - 1:
                assert info["defeat_cause"] is None  # non-terminal step


def test_defense_needed_vs_discarded_recorded():
    env = RegicideGymEnv(EnvConfig(seed=0))
    defense_steps = []
    for seed in range(20):
        infos, _, _ = _collect_episode(env, seed)
        for info in infos:
            if info["defense_needed"] is not None:
                assert info["phase"] == "defense"
                assert info["defense_overspend"] == (
                    info["defense_discarded_sum"] - info["defense_needed"]
                )
                defense_steps.append(info)
    assert defense_steps, "no defense steps encountered across seeds"
    # Successful defenses cover the needed value; the R1 auto-loss (the only
    # possible shortfall) discards 0 and terminates the episode.
    for info in defense_steps:
        if info["result"] != "player_defeated":
            assert info["defense_discarded_sum"] >= info["defense_needed"]


def test_defense_needed_cross_checked_against_engine():
    env = RegicideGymEnv(EnvConfig(seed=0))
    checked = 0
    for seed in range(100):
        obs, info = env.reset(seed=seed)
        mask = info["mask"]
        singles = [a for a in np.flatnonzero(mask) if 1 <= a <= 7]
        if not singles:
            continue
        obs, _, terminated, truncated, info = env.step(int(singles[0]))
        if terminated or truncated or info["result"] != "defense_required":
            continue
        game_state = env.engine.game_state
        hand = list(game_state.get_active_player().hand)
        needed = game_state.current_enemy.effective_attack()
        action = int(np.flatnonzero(info["mask"])[0])
        cards = decode_action(action, hand, "defense", needed)["cards"]
        obs, _, terminated, truncated, info = env.step(action)
        assert info["defense_needed"] == needed
        assert info["defense_discarded_sum"] == sum(c.value for c in cards)
        assert info["defense_overspend"] == sum(c.value for c in cards) - needed
        checked += 1
        if checked >= 5:
            break
    assert checked >= 1, "never reached a defense step to cross-check"


def test_defeat_cause_recorded_on_terminal():
    env = RegicideGymEnv(EnvConfig(seed=0))
    seen = set()
    for seed in range(30):
        for policy in ("random", "zero_single"):
            infos, terminated, truncated = _collect_episode(env, seed, policy=policy)
            final = infos[-1]
            assert final["defeat_cause"] in DEFEAT_CAUSES
            if truncated:
                assert final["defeat_cause"] == "truncated"
            elif final["result"] == "win":
                assert final["defeat_cause"] in ("win", "exact_win")
            elif final["result"] == "player_defeated":
                assert final["defeat_cause"] in (
                    "uncoverable_defense",
                    "insufficient_defense",
                    "stuck",
                )
            seen.add(final["defeat_cause"])
    assert seen, "no terminal episodes observed"


def test_damage_dealt_matches_enemy_hp_drop():
    """Sanity: on a scripted single-card play, damage_dealt equals the enemy
    hp drop (hp remaining on a defeating play — the enemy goes to 0 hp)."""
    env = RegicideGymEnv(EnvConfig(seed=0))
    checked = 0
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        mask = info["mask"]
        singles = [a for a in np.flatnonzero(mask) if 1 <= a <= 7]
        if not singles:
            continue
        enemy = env.engine.game_state.current_enemy
        pre_hp = enemy.max_health - enemy.current_damage
        obs, _, terminated, truncated, info = env.step(int(singles[0]))
        if info["result"] in ("enemy_defeated", "win"):
            hp_drop = pre_hp  # enemy replaced; its hp went to exactly 0
        else:
            assert env.engine.game_state.current_enemy is enemy
            hp_drop = pre_hp - (enemy.max_health - enemy.current_damage)
        assert info["damage_dealt"] == hp_drop
        checked += 1
        if checked >= 10:
            break
    assert checked >= 1, "never played a single card to check damage"


# --------------------------------------------------------------------- #
# Step 3.2 — callback persistence + TensorBoard
# --------------------------------------------------------------------- #
TELEMETRY_RUN_KWARGS = dict(
    campaign_length_schedule=[1],
    steps=2000,
    n_envs=2,
    seed=0,
    eval_interval=1000,
    n_eval_episodes=2,
)


@pytest.fixture(scope="module")
def telemetry_run(tmp_path_factory):
    config = TrainingConfig(
        artifact_dir=str(tmp_path_factory.mktemp("telemetry_run")),
        **TELEMETRY_RUN_KWARGS,
    )
    return train(config)


def _read_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_per_step_jsonl_written_with_fields(telemetry_run):
    path = Path(telemetry_run["steps_jsonl"])
    assert path.is_file()
    records = _read_jsonl(path)
    assert records
    required = {
        "timestep",
        "env",
        "episode",
        "phase",
        "action_idx",
        "decoded_cards",
        "damage_dealt",
        "shield_raised",
        "suit_powers",
        "yield_taken",
        "defense_needed",
        "defense_discarded_sum",
        "defense_overspend",
        "reward",
    }
    for rec in records:
        assert required <= set(rec)
        assert rec["reward"] is not None
    timesteps = [rec["timestep"] for rec in records]
    assert timesteps == sorted(timesteps)  # monotonic


def test_episode_scalars_written(telemetry_run):
    run_dir = Path(telemetry_run["run_dir"])
    episodes = _read_jsonl(run_dir / "episodes.jsonl")
    steps = _read_jsonl(run_dir / "steps.jsonl")
    assert episodes
    required = {
        "timestep",
        "env",
        "win",
        "loss",
        "result",
        "length",
        "enemies_defeated",
        "damage_per_turn",
        "yield_rate",
        "discard_overspend_mean",
        "campaign_length",
    }
    for rec in episodes:
        assert required <= set(rec)
        assert isinstance(rec["win"], bool) and isinstance(rec["loss"], bool)
        assert not (rec["win"] and rec["loss"])
        assert rec["result"] in ("win", "player_defeated", "truncated")
        assert rec["length"] >= 1
        assert 0.0 <= rec["yield_rate"] <= 1.0
        assert rec["damage_per_turn"] >= 0
        assert rec["enemies_defeated"] >= 0
    # Consistency with steps.jsonl aggregation: per env, the k-th episode
    # record's length equals the number of steps.jsonl lines for
    # (env, episode=k).
    step_counts = Counter((rec["env"], rec["episode"]) for rec in steps)
    by_env = {}
    for rec in episodes:
        by_env.setdefault(rec["env"], []).append(rec)
    for env_id, recs in by_env.items():
        for ep_idx, rec in enumerate(recs):
            assert step_counts.get((env_id, ep_idx), 0) == rec["length"]


def test_tensorboard_events_written(telemetry_run):
    # SB3 writes events under a <Algo>_<n>/ subdir of tensorboard_log.
    events = list(Path(telemetry_run["run_dir"]).rglob("events.out.tfevents.*"))
    assert events, "no TensorBoard event files in run dir"
