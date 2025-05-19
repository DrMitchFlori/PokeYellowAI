"""Gym-compatible environment for raw ROM observations."""
from typing import Any, Tuple

import gym

from .emulator import Emulator


class RomEnv(gym.Env):
    """Gym-compatible environment exposing raw emulator memory."""

    def __init__(self, game: str, retro_dir: str | None = None, **make_kwargs: Any) -> None:
        super().__init__()
        self.emulator = Emulator(game, retro_dir, **make_kwargs)
        self.action_space = self.emulator.action_space
        self.observation_space = self.emulator.observation_space

    def reset(self, **kwargs: Any):
        obs = self.emulator.reset()
        return obs

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        return self.emulator.step(action)

    def get_ram(self) -> bytes:
        return self.emulator.get_ram()

    def close(self) -> None:
        self.emulator.close()
        super().close()
