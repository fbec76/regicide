"""Evaluate PPO checkpoints on a fixed campaign length (Step 2.3, deliverable 4).

Loads final_model.zip, best_model.zip, and every checkpoints/*.zip from a
proof run dir, evaluates each with the shared harness
(common.evaluate_policy: fresh seeded env per episode, reset(seed=seed_base+i),
deterministic=True per M5) on campaign_length=3 for 200 episodes.

Paired-seed design: ALL checkpoints use the SAME seed_base so checkpoint-to-
checkpoint and checkpoint-vs-baseline comparisons share the 200 episodes
(lower variance for the first-half vs second-half trend). seed_base is chosen
disjoint from the baseline measurement ranges (pair_index * 10_000 + L * 1_000,
max episode seed ~62_000) so no eval episode reuses a baseline seed.

Output: <run_dir>/eval_report.json + a step -> win_rate/Wilson-LB table on
stdout. (ml/runs/ is gitignored — the report stays out of git.)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.trainers.common import evaluate_policy, wilson_lower_bound

EPISODES = 200
CAMPAIGN_LENGTH = 3
DEFAULT_SEED_BASE = 1_000_000  # disjoint from baseline seed ranges (< ~62k)

_STEP_RE = re.compile(r"ppo_model_(\d+)_steps\.zip$")


def _load_policy(path: Path):
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(str(path), device="cpu")


def evaluate_checkpoint(path: Path, campaign_length: int, n_episodes: int, seed_base: int) -> dict:
    """One model -> win-rate record (+ Wilson LB) via the shared harness."""
    model = _load_policy(path)
    result = evaluate_policy(model, campaign_length, n_episodes, seed_base, deterministic=True)
    record = {
        "win_rate": result["win_rate"],
        "wins": result["wins"],
        "episodes": result["episodes"],
        "wilson_lb": wilson_lower_bound(result["wins"], result["episodes"]),
    }
    del model  # free the policy before the next load
    return record


def evaluate_run(run_dir: Path, campaign_length: int, n_episodes: int, seed_base: int) -> dict:
    """Evaluate final + best + every numbered checkpoint; ordered report."""
    checkpoints: list = []
    ckpt_dir = run_dir / "checkpoints"
    if ckpt_dir.is_dir():
        for path in sorted(ckpt_dir.glob("*.zip")):
            match = _STEP_RE.search(path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
    checkpoints.sort(key=lambda item: item[0])

    entries: list = []
    named = [("best_model", run_dir / "best_model.zip"), ("final_model", run_dir / "final_model.zip")]
    labeled = [(f"checkpoint_{step}", path) for step, path in checkpoints] + [
        (label, path) for label, path in named if path.is_file()
    ]
    for label, path in labeled:
        record = evaluate_checkpoint(path, campaign_length, n_episodes, seed_base)
        entries.append({"model": label, "path": str(path), **record})
        print(f"{label:20s} win_rate={record['win_rate']:.3f} wilson_lb={record['wilson_lb']:.3f} ({record['wins']}/{record['episodes']})")

    report = {
        "meta": {
            "run_dir": str(run_dir),
            "campaign_length": campaign_length,
            "episodes": n_episodes,
            "seed_base": seed_base,
            "deterministic": True,
            "paired_seeds": "all models share seed_base (episodes reset(seed_base+i))",
        },
        "results": entries,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate PPO run checkpoints on a fixed campaign length (Step 2.3)"
    )
    parser.add_argument("--run", type=Path, required=True, help="Proof-run directory (contains final_model.zip / checkpoints/)")
    parser.add_argument("--campaign-length", type=int, default=CAMPAIGN_LENGTH)
    parser.add_argument("--episodes", type=int, default=EPISODES)
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE)
    args = parser.parse_args()

    if not args.run.is_dir():
        raise SystemExit(f"run dir not found: {args.run}")

    report = evaluate_run(args.run, args.campaign_length, args.episodes, args.seed_base)
    out_path = args.run / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")

    header = f"{'model':20s} {'win_rate':>9s} {'wilson_lb':>10s}"
    print(header)
    print("-" * len(header))
    for entry in report["results"]:
        print(f"{entry['model']:20s} {entry['win_rate']:>9.3f} {entry['wilson_lb']:>10.3f}")


if __name__ == "__main__":
    main()
