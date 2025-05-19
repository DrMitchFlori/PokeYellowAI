"""Goal-based reward computation utilities."""

from typing import Iterable, List, Tuple

from types_shared import Goal

from poke_rewards import check_goals


class Rewarder:
    """Compute shaped rewards from WRAM snapshots."""

    def __init__(self, goals: Iterable[Goal]):
        self._goals: List[Goal] = [dict(goal) for goal in goals]

    def compute(
        self, prev_mem: bytes, curr_mem: bytes, env_reward: float = 0.0
    ) -> Tuple[float, List[str]]:
        """Return total reward and IDs of triggered goals."""
        triggered_pairs = check_goals(prev_mem, curr_mem, self._goals)
        total = env_reward + sum(rew for _, rew in triggered_pairs)
        triggered_ids = [gid for gid, _ in triggered_pairs]
        return total, triggered_ids
