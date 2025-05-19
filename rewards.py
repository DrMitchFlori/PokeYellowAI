"""Simple reward computation for PokeYellow gym training.

This module exposes ``compute_reward`` which inspects info dictionaries from
Gym Retro and computes a numeric reward. Adjust the logic as needed.
"""

from typing import Any, Dict


def compute_reward(info: Dict[str, Any]) -> float:
    """Return a basic reward based on the environment info dict.

    Parameters
    ----------
    info: dict
        The info dictionary returned by ``env.step`` in Gym Retro. It is
        expected to contain at least ``score``. You can extend this function
        to incorporate more sophisticated signals as desired.
    """
    # Default reward uses the score field if present; otherwise zero.
    return float(info.get('score', 0))

