"""Sweep runner (Step 4.1): enumerate -> execute (sequential, resumable).

Runs land under ``ml/sweeps/<sweep_id>/<config_name>/seed<k>/`` (the trainer
nests its own run_name/timestamp dir beneath that artifact_dir). Each seed
dir is the unit of resume: on success the trainer's result dict is written to
``run_result.json``; on failure ``run_error.json`` records the error and the
runner continues with the next run. The sweep manifest
(``ml/sweeps/<sweep_id>/manifest.json``) lists every run and its status and
is rewritten after each run so an interrupted sweep can be resumed.

Usage:
    python -m ml.scripts.sweep --dry-run
    python -m ml.scripts.sweep --sweep-id starter-v1
    python -m ml.scripts.sweep --configs aggressive_shaping_ladder_1_3 --seeds 0
    python -m ml.scripts.sweep --max-runs 3
"""
from __future__ import annotations

import argparse
import json
import sys
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from ml.configs.base import RunConfig, RunSpec
from ml.configs.sweep_starter import (
    STARTER_CONFIGS,
    STARTER_SEEDS,
    STARTER_SWEEP_ID,
    config_with_seed,
)
from ml.trainers.ppo_trainer import train

REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ROOT = REPO_ROOT / "ml" / "sweeps"

RUN_RESULT_FILE = "run_result.json"
RUN_ERROR_FILE = "run_error.json"
MANIFEST_FILE = "manifest.json"

STATUS_PENDING = "pending"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"


def run_dir_for(sweep_root: Path, sweep_id: str, config_name: str, seed: int) -> Path:
    return sweep_root / sweep_id / config_name / f"seed{seed}"


def enumerate_runs(
    configs: Dict[str, RunConfig], seeds: List[int], sweep_id: str, sweep_root: Path = SWEEP_ROOT
) -> List[RunSpec]:
    """One run per (config, seed), in config-then-seed order."""
    return [
        RunSpec(
            config=config_with_seed(config, seed),
            seed=seed,
            run_dir=str(run_dir_for(sweep_root, sweep_id, config.name, seed)),
        )
        for config in configs.values()
        for seed in seeds
    ]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_manifest(sweep_id: str, sweep_root: Path, runs: List[RunSpec]) -> dict:
    """Create or refresh the manifest; run dirs with run_result.json are
    recognized as completed even if the manifest write was interrupted."""
    path = sweep_root / sweep_id / MANIFEST_FILE
    existing: Dict[str, dict] = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text())
            existing = {entry["run_dir"]: entry for entry in data.get("runs", [])}
        except (json.JSONDecodeError, KeyError):
            existing = {}

    entries = []
    for run in runs:
        prev = existing.get(run.run_dir, {})
        if Path(run.run_dir, RUN_RESULT_FILE).is_file():
            status = STATUS_COMPLETED
        else:
            status = prev.get("status", STATUS_PENDING)
        entries.append(
            {
                "config_name": run.config_name,
                "seed": run.seed,
                "run_dir": run.run_dir,
                "status": status,
                "error": prev.get("error"),
            }
        )
    return {"sweep_id": sweep_id, "created": _utc_now(), "updated": _utc_now(), "runs": entries}


def _write_manifest(sweep_id: str, sweep_root: Path, manifest: dict) -> None:
    path = sweep_root / sweep_id / MANIFEST_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest["updated"] = _utc_now()
    path.write_text(json.dumps(manifest, indent=2))


def is_completed(run_dir: str) -> bool:
    """A seed dir with run_result.json is a finished run (resume marker)."""
    return Path(run_dir, RUN_RESULT_FILE).is_file()


def execute_sweep(
    runs: List[RunSpec],
    sweep_id: str,
    sweep_root: Path = SWEEP_ROOT,
    max_runs: int = 0,
    train_fn: Callable[[object], dict] = train,
) -> dict:
    """Sequential, resumable execution; one failed run never stops the sweep.

    Returns a summary: {completed, skipped, failed, remaining}.
    """
    manifest = _load_manifest(sweep_id, sweep_root, runs)
    status_by_dir = {entry["run_dir"]: entry for entry in manifest["runs"]}

    summary = {"completed": 0, "skipped": 0, "failed": 0, "remaining": 0}
    executed = 0
    for run in runs:
        entry = status_by_dir[run.run_dir]
        if entry["status"] == STATUS_COMPLETED or is_completed(run.run_dir):
            entry["status"] = STATUS_COMPLETED
            print(f"SKIP (completed) {run.config_name} seed={run.seed}")
            summary["skipped"] += 1
            continue
        if max_runs and executed >= max_runs:
            summary["remaining"] += 1
            continue
        executed += 1
        print(f"RUN {run.config_name} seed={run.seed} -> {run.run_dir}")
        Path(run.run_dir).mkdir(parents=True, exist_ok=True)
        try:
            result = train_fn(run)
        except Exception as error:  # noqa: BLE001 — record and continue
            entry["status"] = STATUS_FAILED
            entry["error"] = f"{type(error).__name__}: {error}"
            Path(run.run_dir, RUN_ERROR_FILE).write_text(
                json.dumps({"error": entry["error"], "traceback": traceback.format_exc()}, indent=2)
            )
            print(f"FAILED {run.config_name} seed={run.seed}: {entry['error']}")
            summary["failed"] += 1
        else:
            entry["status"] = STATUS_COMPLETED
            entry["error"] = None
            Path(run.run_dir, RUN_RESULT_FILE).write_text(json.dumps(result, indent=2))
            print(f"DONE {run.config_name} seed={run.seed}")
            summary["completed"] += 1
        _write_manifest(sweep_id, sweep_root, manifest)

    _write_manifest(sweep_id, sweep_root, manifest)
    return summary


def _print_dry_run(runs: List[RunSpec]) -> None:
    for i, run in enumerate(runs, start=1):
        print(f"[{i:2d}/ {len(runs)}] {run.config_name} seed={run.seed} {run.run_dir}")
    print(f"dry-run: {len(runs)} runs total")


def _parse_int_list(raw: str) -> List[int]:
    return [int(part) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a training sweep (sequential, resumable)"
    )
    parser.add_argument(
        "--sweep-id", default=STARTER_SWEEP_ID, help=f"Sweep id (default: {STARTER_SWEEP_ID})"
    )
    parser.add_argument(
        "--configs",
        default=None,
        help="Comma-separated config-name subset (default: all 5 starter configs)",
    )
    parser.add_argument(
        "--seeds", default=None, help="Comma-separated seed subset (default: 0,1,2)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List the runs and exit without training"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose seed dir already contains run_result.json "
        "(this is the default, safe behavior; the flag is explicit confirmation)",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Execute at most N pending runs this invocation (0 = all)",
    )
    args = parser.parse_args()

    configs = STARTER_CONFIGS
    if args.configs:
        from ml.configs.sweep_starter import select_configs

        configs = select_configs([name.strip() for name in args.configs.split(",")])
    seeds = STARTER_SEEDS if args.seeds is None else _parse_int_list(args.seeds)
    if not seeds:
        raise SystemExit("--seeds must be a comma-separated list, e.g. 0,1,2")

    runs = enumerate_runs(configs, seeds, args.sweep_id)
    if args.dry_run:
        _print_dry_run(runs)
        return

    summary = execute_sweep(runs, args.sweep_id, max_runs=args.max_runs)
    print(
        "sweep {sweep_id}: completed={completed} skipped={skipped} "
        "failed={failed} remaining={remaining}".format(sweep_id=args.sweep_id, **summary)
    )


if __name__ == "__main__":
    main()
