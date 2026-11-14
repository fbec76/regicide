"""Step 4.2 eval harness: per-rung win rates, per-turn behavior stats,
baseline comparison, and the round-robin joint win-rate matrix.

Single-run mode (default):
    python -m ml.scripts.eval --run <run_dir> --episodes N
Evaluates final_model.zip (fallback best_model.zip) per curriculum rung
(default: the run's schedule + the full 12) via the shared harness
(common.evaluate_policy: policy plays BOTH seats, deterministic=True, fresh
seeded env per episode, reset(seed=seed_base+i) per M5). A separate
stats-collection pass replays the SAME episodes (same seeds) with the Phase-3
telemetry infos and aggregates per-turn behavior stats by campaign position
(enemy index 0..L-1), implementing the plan's Sweep-axis metric column:
damage/turn by position, shield timing (enemy hp fraction when raised),
shield frequency, yield rate, suit-power usage counts, defense discard
efficiency (overspend mean + distribution), and hand value retained at
episode end / at win. Output: <run_dir>/eval_report.json + printed tables.

--compare-baselines joins ml/runs/baselines/baselines.json pair win rates per
rung and prints the Phase-2.3-style gate verdict (PPO strictly above the best
baseline AND Wilson LB above the baseline point estimate).

--mode roundrobin (Phase 6 prep): pairwise JOINT win rates for a pool of
models (>= 2 --run dirs; each model plays one seat, partner from the pool).
Each unordered pair is evaluated at BOTH seatings (a@A+b@B and b@A+a@B) and
the joint win rate is their mean, so the matrix is seating-order-insensitive
and symmetric; a model paired with itself (diagonal) is a single seating.
Shape: pairs["nameA+nameB"] (names sorted, deterministic) -> {joint_win_rate,
seatings{...}}; lookup with either name order resolves to the same entry.
Written to eval_matrix.json (cwd default, --out to override).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.envs.regicide_env import EnvConfig
from ml.trainers.common import evaluate_policy, make_env, wilson_lower_bound

DEFAULT_EPISODES = 200
DEFAULT_SEED_BASE = 1_000_000  # disjoint from the baseline seed ranges (< ~62k)
FULL_CAMPAIGN_LENGTH = 12
SEAT_INDEX_BY_NAME = {"Player1": 0, "Player2": 1}
RUN_RESULT_FILE = "run_result.json"
_RUN_NAME_RE = re.compile(r"_L(\d+(?:-\d+)*)_s")
_SUIT_STATUSES = ("fires", "blocked", "pending")


# --------------------------------------------------------------------- #
# Model + rung discovery
# --------------------------------------------------------------------- #

def load_model(run_dir: Path):
    """Load final_model.zip (preferred) or best_model.zip from a run dir."""
    from sb3_contrib import MaskablePPO

    for name in ("final_model.zip", "best_model.zip"):
        path = run_dir / name
        if path.is_file():
            return MaskablePPO.load(str(path), device="cpu"), path
    raise SystemExit(f"no final_model.zip or best_model.zip in {run_dir}")


def discover_schedule(run_dir: Path) -> Optional[list]:
    """Recover the run's campaign_length schedule (curriculum ladder).

    Source of truth is run_result.json (sweep runs); fallback parses the
    trainer's run-name convention <algo>_L<1-3>_s<seed> on the parent dir.
    """
    result_file = run_dir / RUN_RESULT_FILE
    if result_file.is_file():
        try:
            data = json.loads(result_file.read_text())
        except json.JSONDecodeError:
            data = None
        schedule = (data or {}).get("campaign_length_schedule")
        if (
            isinstance(schedule, list)
            and schedule
            and all(isinstance(v, int) and v > 0 for v in schedule)
        ):
            return list(schedule)
    match = _RUN_NAME_RE.search(run_dir.parent.name)
    if match:
        return [int(part) for part in match.group(1).split("-")]
    return None


def default_eval_rungs(schedule: Optional[list]) -> list:
    """Eval rungs: the run's curriculum rungs + the full 12."""
    rungs = {FULL_CAMPAIGN_LENGTH}
    if schedule:
        rungs.update(schedule)
    return sorted(rungs)


# --------------------------------------------------------------------- #
# Per-turn behavior stats (Sweep-axis metric column)
# --------------------------------------------------------------------- #

def run_stats_episode(model: Any, campaign_length: int, seed: int) -> dict:
    """One episode with the policy in both seats, collecting per-step records.

    Same seeding as common.evaluate_policy (reset(seed=seed_base+i)) so the
    stats pass replays exactly the episodes the win-rate pass scored. Pre-step
    the enemy index (L-1-castle_deck remaining: the engine draws the next
    enemy from castle_deck on defeat) and the enemy's remaining hp fraction
    are captured read-only from the engine game state.
    """
    env = make_env(EnvConfig(campaign_length=campaign_length), seed=seed)
    obs, info = env.reset(seed=seed)
    steps: list = []
    terminated = truncated = False
    while not (terminated or truncated):
        game_state = env.engine.game_state
        enemy = game_state.current_enemy
        position = campaign_length - 1 - len(game_state.castle_deck.cards)
        hp_fraction = 1.0 - enemy.current_damage / enemy.max_health
        action, _ = model.predict(
            obs, action_masks=info["mask"], deterministic=True
        )
        obs, _, terminated, truncated, info = env.step(int(action))
        steps.append(
            {
                "position": position,
                "damage_dealt": int(info["damage_dealt"]),
                "shield_raised": int(info["shield_raised"]),
                "enemy_hp_fraction": hp_fraction,
                "yield_taken": bool(info["yield_taken"]),
                "suit_powers": dict(info["suit_powers"]),
                "defense_overspend": info["defense_overspend"],
            }
        )
    win = bool(terminated and info.get("result") == "win")
    hand_value_end = sum(
        card.value
        for player in env.engine.game_state.players
        for card in player.hand
    )
    env.close()
    return {"steps": steps, "win": win, "hand_value_end": hand_value_end}


def _mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _position_stats(acc: dict) -> dict:
    n = acc["steps"]
    overspends = acc["overspends"]
    dist = Counter(overspends)
    return {
        "steps": n,
        "damage_per_turn": acc["damage_sum"] / n,
        "shield_rate": acc["shields"] / n,
        "shield_hp_fraction_mean": _mean(acc["shield_hp"]),
        "yield_rate": acc["yields"] / n,
        "suit_powers": {
            suit: {status: counts.get(status, 0) for status in _SUIT_STATUSES}
            for suit, counts in acc["suits"].items()
        },
        "defense_steps": len(overspends),
        "defense_overspend_mean": _mean(overspends),
        "defense_overspend_dist": {
            str(value): count for value, count in sorted(dist.items())
        },
    }


def aggregate_behavior_stats(episodes: list) -> dict:
    """Aggregate stats-episode records by campaign position + episode level."""
    per_position: dict = {}
    hand_values: list = []
    win_hand_values: list = []
    wins = 0
    for episode in episodes:
        wins += episode["win"]
        hand_values.append(episode["hand_value_end"])
        if episode["win"]:
            win_hand_values.append(episode["hand_value_end"])
        for rec in episode["steps"]:
            acc = per_position.get(rec["position"])
            if acc is None:
                acc = per_position[rec["position"]] = {
                    "steps": 0,
                    "damage_sum": 0,
                    "shields": 0,
                    "shield_hp": [],
                    "yields": 0,
                    "suits": {},
                    "overspends": [],
                }
            acc["steps"] += 1
            acc["damage_sum"] += rec["damage_dealt"]
            if rec["shield_raised"] > 0:
                acc["shields"] += 1
                acc["shield_hp"].append(rec["enemy_hp_fraction"])
            if rec["yield_taken"]:
                acc["yields"] += 1
            for suit, status in rec["suit_powers"].items():
                acc["suits"].setdefault(suit, Counter())[status] += 1
            if rec["defense_overspend"] is not None:
                acc["overspends"].append(rec["defense_overspend"])
    return {
        "by_position": {
            str(position): _position_stats(acc)
            for position, acc in sorted(per_position.items())
        },
        "episode": {
            "episodes": len(episodes),
            "wins": wins,
            "hand_value_end_mean": _mean(hand_values),
            "hand_value_end_win_mean": (
                _mean(win_hand_values) if win_hand_values else None
            ),
        },
    }


# --------------------------------------------------------------------- #
# Per-rung evaluation + baseline gate
# --------------------------------------------------------------------- #

def evaluate_rung(model: Any, campaign_length: int, n_episodes: int, seed_base: int) -> dict:
    """Win rate (shared harness) + behavior stats over the same episodes."""
    result = evaluate_policy(
        model, campaign_length, n_episodes, seed_base, deterministic=True
    )
    stats_episodes = [
        run_stats_episode(model, campaign_length, seed_base + i)
        for i in range(n_episodes)
    ]
    return {
        "win_rate": result["win_rate"],
        "wins": result["wins"],
        "episodes": result["episodes"],
        "wilson_lb": wilson_lower_bound(result["wins"], result["episodes"]),
        "behavior": aggregate_behavior_stats(stats_episodes),
    }


def load_baselines(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"baselines file not found: {path}")
    data = json.loads(path.read_text())
    if "pairs" not in data:
        raise SystemExit(f"baselines file has no 'pairs' key: {path}")
    return data


def best_baseline(baselines: dict, campaign_length: int) -> Optional[dict]:
    """Best scripted-pair win rate measured at this campaign length."""
    best = None
    for pair, rungs in baselines["pairs"].items():
        record = rungs.get(str(campaign_length))
        if record and (best is None or record["win_rate"] > best["record"]["win_rate"]):
            best = {"pair": pair, "record": record}
    return best


def gate_verdict(win_rate: float, wilson_lb: float, baseline_win_rate: float) -> str:
    """Phase-2.3 gate pattern: strictly exceeds AND Wilson LB above baseline."""
    if win_rate > baseline_win_rate and wilson_lb > baseline_win_rate:
        return "PPO_EXCEEDS_BASELINE"
    return "PPO_BELOW_BASELINE"


def build_report(
    run_dir: Path,
    rungs: list,
    n_episodes: int,
    seed_base: int,
    baselines: Optional[dict] = None,
) -> dict:
    """One run -> per-rung win rates + behavior stats (+ baseline joins)."""
    model, model_path = load_model(run_dir)
    rung_records: dict = {}
    baseline_records: dict = {}
    for length in rungs:
        record = evaluate_rung(model, length, n_episodes, seed_base)
        rung_records[str(length)] = record
        if baselines is not None:
            best = best_baseline(baselines, length)
            baseline_records[str(length)] = (
                {
                    "pair": best["pair"],
                    "baseline_win_rate": best["record"]["win_rate"],
                    "verdict": gate_verdict(
                        record["win_rate"],
                        record["wilson_lb"],
                        best["record"]["win_rate"],
                    ),
                }
                if best
                else None
            )
    del model
    report = {
        "meta": {
            "run_dir": str(run_dir),
            "model": str(model_path),
            "campaign_lengths": list(rungs),
            "episodes": n_episodes,
            "seed_base": seed_base,
            "deterministic": True,
            "seed_convention": "episode i resets env with seed seed_base+i",
        },
        "rungs": rung_records,
    }
    if baselines is not None:
        report["baselines"] = baseline_records
    return report


# --------------------------------------------------------------------- #
# Round-robin joint win-rate matrix (Phase 6 consumer)
# --------------------------------------------------------------------- #

def model_label(run_dir: Path) -> str:
    return f"{run_dir.parent.name}/{run_dir.name}"


def pair_key(name_a: str, name_b: str) -> str:
    return "+".join(sorted([name_a, name_b]))


def run_joint_episode(policies: list, campaign_length: int, seed: int) -> bool:
    """One episode with one policy per seat; True iff the team wins."""
    env = make_env(EnvConfig(campaign_length=campaign_length), seed=seed)
    obs, info = env.reset(seed=seed)
    terminated = truncated = False
    while not (terminated or truncated):
        active = env.engine.game_state.get_active_player().name
        seat = SEAT_INDEX_BY_NAME.get(active)
        if seat is None:
            raise ValueError(
                f"unsupported seat name for round-robin: {active!r} "
                "(round-robin is 2-player only in Phase 4)"
            )
        action, _ = policies[seat].predict(
            obs, action_masks=info["mask"], deterministic=True
        )
        obs, _, terminated, truncated, info = env.step(int(action))
    win = bool(terminated and info.get("result") == "win")
    env.close()
    return win


def _seating_rate(policies: list, campaign_length: int, n_episodes: int, seed_base: int) -> dict:
    wins = sum(
        run_joint_episode(policies, campaign_length, seed_base + i)
        for i in range(n_episodes)
    )
    return {"win_rate": wins / n_episodes, "wins": wins, "episodes": n_episodes}


def evaluate_pair(
    model_a: Any,
    name_a: str,
    model_b: Any,
    name_b: str,
    campaign_length: int,
    n_episodes: int,
    seed_base: int,
) -> dict:
    """Joint win rate for one pair, averaged over both seatings (symmetric).

    A model paired with itself (diagonal) is a single seating.
    """
    if name_a == name_b:
        seating = _seating_rate(
            [model_a, model_b], campaign_length, n_episodes, seed_base
        )
        return {
            "joint_win_rate": seating["win_rate"],
            "seatings": {f"{name_a}@A+{name_b}@B": seating},
        }
    forward = _seating_rate([model_a, model_b], campaign_length, n_episodes, seed_base)
    reverse = _seating_rate([model_b, model_a], campaign_length, n_episodes, seed_base)
    return {
        "joint_win_rate": (forward["win_rate"] + reverse["win_rate"]) / 2.0,
        "seatings": {
            f"{name_a}@A+{name_b}@B": forward,
            f"{name_b}@A+{name_a}@B": reverse,
        },
    }


def evaluate_matrix(
    models: list,
    campaign_length: int,
    n_episodes: int,
    seed_base: int,
) -> dict:
    """All unordered pairs (incl. diagonal) of a model pool -> matrix."""
    labels = [name for name, _ in models]
    if len(set(labels)) != len(labels):
        raise SystemExit(f"duplicate run labels in pool: {labels}")
    pairs: dict = {}
    for i in range(len(models)):
        for j in range(i, len(models)):
            (name_a, model_a), (name_b, model_b) = models[i], models[j]
            key = pair_key(name_a, name_b)
            pairs[key] = evaluate_pair(
                model_a, name_a, model_b, name_b, campaign_length, n_episodes, seed_base
            )
    return {
        "meta": {
            "campaign_length": campaign_length,
            "episodes_per_seating": n_episodes,
            "seed_base": seed_base,
            "deterministic": True,
            "shape": (
                "symmetric; keys 'nameA+nameB' with names sorted "
                "(pair_key(a, b) == pair_key(b, a)); joint_win_rate is the "
                "mean over both seatings (diagonal self-pairs: one seating); "
                "raw per-seating rates under 'seatings'"
            ),
        },
        "models": labels,
        "pairs": pairs,
    }


def run_roundrobin(
    run_dirs: list,
    campaign_length: int,
    n_episodes: int,
    seed_base: int,
    out_path: Path,
) -> dict:
    models = [(model_label(run_dir), load_model(run_dir)[0]) for run_dir in run_dirs]
    matrix = evaluate_matrix(models, campaign_length, n_episodes, seed_base)
    out_path.write_text(json.dumps(matrix, indent=2) + "\n")
    return matrix


# --------------------------------------------------------------------- #
# Printing
# --------------------------------------------------------------------- #

def _fmt(value: Optional[float], spec: str = ".3f") -> str:
    return "n/a" if value is None else format(value, spec)


def _print_rung_behavior(length: int, record: dict) -> None:
    behavior = record["behavior"]
    header = (
        f"{'pos':>3} {'steps':>6} {'dmg/turn':>9} {'shield%':>8} "
        f"{'shield@hp':>10} {'yield%':>7} {'ovsp_mean':>10}"
    )
    print(f"\nbehavior stats — rung {length}")
    print(header)
    for position, stats in behavior["by_position"].items():
        print(
            f"{position:>3} {stats['steps']:>6} {stats['damage_per_turn']:>9.2f} "
            f"{stats['shield_rate']:>8.2%} {_fmt(stats['shield_hp_fraction_mean'], '.2f'):>10} "
            f"{stats['yield_rate']:>7.2%} {_fmt(stats['defense_overspend_mean'], '.2f'):>10}"
        )
    suit_totals: dict = {}
    for stats in behavior["by_position"].values():
        for suit, counts in stats["suit_powers"].items():
            total = suit_totals.setdefault(suit, Counter())
            total.update(counts)
    suit_str = " ".join(
        f"{suit}:{counts['fires']}/{counts['blocked']}/{counts['pending']}"
        for suit, counts in sorted(suit_totals.items())
    )
    episode = behavior["episode"]
    print(
        f"suit_powers (fires/blocked/pending): {suit_str or 'none'}"
    )
    print(
        f"episode: wins={episode['wins']}/{episode['episodes']} "
        f"hand_value_end_mean={_fmt(episode['hand_value_end_mean'], '.1f')} "
        f"hand_value_end_win_mean={_fmt(episode['hand_value_end_win_mean'], '.1f')}"
    )


def print_single_report(report: dict, baselines_path: Optional[Path]) -> None:
    meta = report["meta"]
    print(
        f"== eval run={meta['run_dir']} model={Path(meta['model']).name} "
        f"episodes={meta['episodes']} seed_base={meta['seed_base']} "
        f"rungs={','.join(str(L) for L in meta['campaign_lengths'])} =="
    )
    header = f"{'rung':>4} {'win_rate':>9} {'wilson_lb':>10} {'wins/eps':>9}"
    print(header)
    print("-" * len(header))
    for length, record in report["rungs"].items():
        print(
            f"{length:>4} {record['win_rate']:>9.3f} {record['wilson_lb']:>10.3f} "
            f"{record['wins']}/{record['episodes']}"
        )
    for length, record in report["rungs"].items():
        _print_rung_behavior(int(length), record)
    if baselines_path is not None:
        print(f"\nbaseline comparison ({baselines_path}):")
        print(
            f"{'rung':>4} {'best pair':<34} {'base_rate':>9} {'ppo':>6} "
            f"{'ppo_lb':>7} {'verdict'}"
        )
        for length, comparison in report.get("baselines", {}).items():
            if comparison is None:
                print(f"{length:>4} {'(no baseline measured)':<34} {'n/a':>9}")
                continue
            ppo = report["rungs"][length]
            print(
                f"{length:>4} {comparison['pair']:<34} "
                f"{comparison['baseline_win_rate']:>9.3f} {ppo['win_rate']:>6.3f} "
                f"{ppo['wilson_lb']:>7.3f} {comparison['verdict']}"
            )


def print_roundrobin(matrix: dict) -> None:
    meta = matrix["meta"]
    print(
        f"== round-robin joint win rates: campaign_length={meta['campaign_length']} "
        f"episodes/seating={meta['episodes_per_seating']} seed_base={meta['seed_base']} =="
    )
    labels = matrix["models"]
    width = max(len(label) for label in labels) + 2
    header = " " * width + "".join(f"{label:>{width}}" for label in labels)
    print(header)
    for row in labels:
        cells = [
            f"{matrix['pairs'][pair_key(row, col)]['joint_win_rate']:>{width}.3f}"
            for col in labels
        ]
        print(f"{row:>{width}}" + "".join(cells))
    for key, pair in matrix["pairs"].items():
        seatings = "  ".join(
            f"{seating}={rates['win_rate']:.3f}"
            for seating, rates in pair["seatings"].items()
        )
        print(f"{key}: joint={pair['joint_win_rate']:.3f} ({seatings})")


# --------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------- #

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 4.2 eval harness: per-rung win rates + behavior stats "
        "(--run), baseline comparison (--compare-baselines), round-robin matrix "
        "(--mode roundrobin)"
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        required=True,
        help="Run dir (contains final_model.zip/best_model.zip); repeatable "
        "for --mode roundrobin (>= 2)",
    )
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--campaign-length",
        type=int,
        action="append",
        dest="campaign_lengths",
        help="Rung to evaluate; repeatable. Default: the run's curriculum "
        "rungs + full 12 (single mode) / one length for roundrobin",
    )
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    parser.add_argument(
        "--compare-baselines",
        type=Path,
        help="baselines.json path (single mode): join pair win rates per rung "
        "+ print the gate verdict",
    )
    parser.add_argument(
        "--mode", choices=["single", "roundrobin"], default="single"
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output path; default <run>/eval_report.json (single) or "
        "./eval_matrix.json (roundrobin)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.episodes < 1:
        raise SystemExit(f"--episodes must be >= 1, got {args.episodes}")
    for run_dir in args.run:
        if not run_dir.is_dir():
            raise SystemExit(f"run dir not found: {run_dir}")

    if args.mode == "roundrobin":
        if len(args.run) < 2:
            raise SystemExit("--mode roundrobin requires >= 2 --run dirs")
        if not args.campaign_lengths:
            discovered = [
                L for run_dir in args.run for L in discover_schedule(run_dir) or []
            ]
            lengths = [max(discovered) if discovered else FULL_CAMPAIGN_LENGTH]
        else:
            lengths = sorted(set(args.campaign_lengths))
        if len(lengths) != 1:
            raise SystemExit(
                "--mode roundrobin evaluates ONE campaign length, got "
                f"{lengths}"
            )
        out_path = args.out or Path("eval_matrix.json")
        matrix = run_roundrobin(
            args.run, lengths[0], args.episodes, args.seed_base, out_path
        )
        print_roundrobin(matrix)
        print(f"wrote {out_path}")
        return

    if len(args.run) != 1:
        raise SystemExit("single mode takes exactly one --run")
    run_dir = args.run[0]
    if args.campaign_lengths:
        rungs = sorted(set(args.campaign_lengths))
    else:
        rungs = default_eval_rungs(discover_schedule(run_dir))
    baselines = (
        load_baselines(args.compare_baselines) if args.compare_baselines else None
    )
    report = build_report(run_dir, rungs, args.episodes, args.seed_base, baselines)
    out_path = args.out or run_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print_single_report(report, args.compare_baselines)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
