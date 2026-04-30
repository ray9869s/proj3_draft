"""Placeholder reinforcement learning environment definitions."""


class AirJetEnv:
    """Minimal placeholder environment for future RL experiments."""

    def __init__(self) -> None:
        # TODO: Define observation space, action space, and reset logic.
        self.name = "AirJetEnv"

    def reset(self):
        """Reset the environment state."""
        # TODO: Return the initial observation and info dict.
        return None, {}

    def step(self, action):
        """Advance the environment by one step."""
        # TODO: Apply `action` and return observation, reward, terminated, truncated, info.
        return None, 0.0, True, False, {}
