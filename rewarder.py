"""Goal-based reward computation utilities."""

from typing import Iterable, List, Tuple, Callable

from types_shared import GoalDict

from poke_rewards import (
    _map_changed,
    _badge_bit_set,
    _event_flag_set,
    check_goals,
)


Predicate = Callable[[bytes, bytes], bool]


def predicate_from_goal(goal: GoalDict) -> Predicate:
    """Create a predicate function for a single goal."""
    gtype = goal.get("type")
    target_id = int(goal.get("target_id", 0))

    if gtype == "map":
        def pred(prev: bytes, curr: bytes) -> bool:
            changed, curr_map = _map_changed(prev, curr)
            return changed and curr_map == target_id
        return pred
    elif gtype == "event":
        if target_id < 8:
            def pred(prev: bytes, curr: bytes) -> bool:
                return _badge_bit_set(prev, curr, target_id)
            return pred
        else:
            def pred(prev: bytes, curr: bytes) -> bool:
                return _event_flag_set(prev, curr, target_id - 8)
            return pred
    else:
        raise ValueError(f"Unknown goal type: {gtype}")

class Rewarder:
    """Compute shaped rewards from WRAM snapshots."""

    def __init__(self, goals: Iterable[GoalDict]):
        self._goals: List[GoalDict] = []
        for goal in goals:
            self._goals.append(goal)

    def compute(
        self, prev_mem: bytes, curr_mem: bytes, env_reward: float = 0.0
    ) -> Tuple[float, List[str]]:
        """Return total reward and IDs of triggered goals."""
        triggered_pairs = check_goals(prev_mem, curr_mem, self._goals)
        total = env_reward + sum(rew for _, rew in triggered_pairs)
        triggered_ids = [gid for gid, _ in triggered_pairs]
        return total, triggered_ids
