"""Placeholder entry point for reinforcement learning training."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rl_env import AirJetEnv


def main() -> None:
    # TODO: Connect the environment to Stable-Baselines3 training.
    env = AirJetEnv()
    print(f"RL training placeholder is ready for {env.name}.")


if __name__ == "__main__":
    main()
