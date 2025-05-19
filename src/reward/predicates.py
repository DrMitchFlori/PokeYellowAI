"""Goal predicate implementations for memory-based rewards."""

from dataclasses import dataclass
from typing import Dict

from poke_rewards import MAP_ID_ADDR, BADGE_FLAGS_ADDR, EVENT_FLAGS_BASE


@dataclass
class Predicate:
    """Base class for goal predicates."""

    id: str
    reward: float

    def triggered(self, prev: bytes, curr: bytes) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError


@dataclass
class MapPredicate(Predicate):
    target_id: int

    def triggered(self, prev: bytes, curr: bytes) -> bool:
        prev_id = prev[MAP_ID_ADDR]
        curr_id = curr[MAP_ID_ADDR]
        return prev_id != curr_id and curr_id == self.target_id


def _badge_bit_set(prev: bytes, curr: bytes, bit: int) -> bool:
    mask = 1 << bit
    return not (prev[BADGE_FLAGS_ADDR] & mask) and (curr[BADGE_FLAGS_ADDR] & mask)


def _event_flag_set(prev: bytes, curr: bytes, flag_index: int) -> bool:
    byte_offset = EVENT_FLAGS_BASE + flag_index // 8
    bit = flag_index % 8
    mask = 1 << bit
    return not (prev[byte_offset] & mask) and (curr[byte_offset] & mask)


@dataclass
class EventPredicate(Predicate):
    target_id: int

    def triggered(self, prev: bytes, curr: bytes) -> bool:
        if self.target_id < 8:
            return _badge_bit_set(prev, curr, self.target_id)
        return _event_flag_set(prev, curr, self.target_id - 8)


def predicate_from_goal(goal: Dict[str, object]) -> Predicate:
    """Create a Predicate instance from a goal dictionary."""
    gid = str(goal.get("id"))
    gtype = goal.get("type")
    reward = float(goal.get("reward", 1.0))
    target = int(goal.get("target_id", 0))

    if gtype == "map":
        return MapPredicate(id=gid, reward=reward, target_id=target)
    if gtype == "event":
        return EventPredicate(id=gid, reward=reward, target_id=target)
    raise ValueError(f"Unsupported goal type: {gtype}")
