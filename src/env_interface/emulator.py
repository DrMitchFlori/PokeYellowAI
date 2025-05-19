import retro
from typing import Any, Tuple


class Emulator:
    """Wrapper around ``gym-retro`` environments providing basic emulator access."""

    def __init__(self, game: str, retro_dir: str | None = None, **make_kwargs: Any) -> None:
        if retro_dir:
            retro.data.Integrations.add_custom_path(retro_dir)
        self.env = retro.make(game=game, **make_kwargs)

    # Expose common spaces for gym-style compatibility
    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def reset(self) -> Any:
        """Reset the emulator and return the first observation."""
        return self.env.reset()

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        """Advance the emulator by one frame using the given action."""
        return self.env.step(action)

    def get_ram(self) -> bytes:
        """Return a copy of the emulator's RAM."""
        return bytes(self.env.get_ram())

    def close(self) -> None:
        self.env.close()
