import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse

from ml.trainers.ppo_trainer import TrainingConfig, train


def _parse_schedule(args) -> list:
    """--curriculum "1,3" wins over --campaign-length; default ladder [1, 3]."""
    if args.curriculum:
        schedule = [int(part) for part in args.curriculum.split(",") if part.strip()]
        if not schedule:
            raise SystemExit("--curriculum must be a comma-separated list, e.g. 1,3")
        return schedule
    if args.campaign_length:
        return [args.campaign_length]
    return [1, 3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent to play Regicide")
    parser.add_argument("--algo", default="ppo", choices=["ppo"], help="Algorithm (default: ppo)")
    parser.add_argument("--campaign-length", type=int, default=None, help="Fixed campaign length L (single rung)")
    parser.add_argument("--curriculum", type=str, default=None, help="Comma-separated ladder, e.g. \"1,3\"")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Run-level random seed")
    parser.add_argument("--device", default="cpu", help="torch device (cpu | mps)")
    parser.add_argument("--n-envs", type=int, default=8, help="Parallel DummyVecEnv environments")
    parser.add_argument("--eval-interval", type=int, default=None, help="Eval/checkpoint interval in per-env steps")
    args = parser.parse_args()

    schedule = _parse_schedule(args)
    # Default eval cadence: guarantee >=1 eval/checkpoint mid-run regardless of
    # step count (eval_freq is per-env steps -> n_calls total = steps / n_envs).
    eval_interval = args.eval_interval
    if eval_interval is None:
        eval_interval = max(1, args.steps // (2 * args.n_envs))

    config = TrainingConfig(
        algo=args.algo,
        campaign_length_schedule=schedule,
        steps=args.steps,
        n_envs=args.n_envs,
        seed=args.seed,
        device=args.device,
        eval_interval=eval_interval,
    )
    print(
        f"algo={config.algo} campaign_length_schedule={schedule} "
        f"steps={config.steps} seed={config.seed} device={config.device} "
        f"n_envs={config.n_envs} eval_interval={eval_interval}"
    )
    result = train(config)
    print(f"run_dir={result['run_dir']}")
    print(f"final_model={result['final_model']}")
    print(f"best_model={result['best_model']}")
    print(f"invalid_action_count={result['invalid_action_count']}")


if __name__ == "__main__":
    main()
