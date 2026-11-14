import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an RL agent to play Regicide")
    parser.add_argument("--algo", default="maskable_ppo", help="Algorithm (default: maskable_ppo)")
    parser.add_argument("--campaign-length", type=int, default=6, help="Number of castle enemies L")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    print(f"algo={args.algo} campaign_length={args.campaign_length} "
          f"steps={args.steps} seed={args.seed}")
    # Full trainer wiring arrives in Phase 2.


if __name__ == "__main__":
    main()
