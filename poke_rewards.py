"""Reward detection utilities for Pokémon Yellow WRAM snapshots."""

from typing import Iterable, List, Tuple

from types_shared import GoalDict

# Memory addresses based on community documentation
# Address of the current map ID within WRAM
MAP_ID_ADDR = 0xD35D
BADGE_FLAGS_ADDR = 0xD356
EVENT_FLAGS_BASE = 0xD747


def _map_changed(prev: bytes, curr: bytes) -> Tuple[bool, int]:
    """Return (changed, current map ID)."""
    if len(prev) <= MAP_ID_ADDR:
        raise ValueError(
            f"prev buffer too small; expected index {MAP_ID_ADDR:#x} accessible, got length {len(prev)}"
        )
    if len(curr) <= MAP_ID_ADDR:
        raise ValueError(
            f"curr buffer too small; expected index {MAP_ID_ADDR:#x} accessible, got length {len(curr)}"
        )

    prev_id = prev[MAP_ID_ADDR]
    curr_id = curr[MAP_ID_ADDR]
    return (prev_id != curr_id, curr_id)


def _badge_bit_set(prev: bytes, curr: bytes, bit: int) -> bool:
    """Return True if the given badge bit transitioned from 0 to 1."""
    if len(prev) <= BADGE_FLAGS_ADDR:
        raise ValueError(
            f"prev buffer too small; expected index {BADGE_FLAGS_ADDR:#x} accessible, got length {len(prev)}"
        )
    if len(curr) <= BADGE_FLAGS_ADDR:
        raise ValueError(
            f"curr buffer too small; expected index {BADGE_FLAGS_ADDR:#x} accessible, got length {len(curr)}"
        )

    mask = 1 << bit
    return not (prev[BADGE_FLAGS_ADDR] & mask) and (curr[BADGE_FLAGS_ADDR] & mask)


def _event_flag_set(prev: bytes, curr: bytes, flag_index: int) -> bool:
    """Return True if a generic event flag bit transitioned from 0 to 1."""
    if flag_index < 0:
        raise ValueError(f"flag_index must be non-negative, got {flag_index}")

    byte_offset = EVENT_FLAGS_BASE + flag_index // 8
    if len(prev) <= byte_offset:
        raise ValueError(
            "prev buffer too small for event flag index "
            f"{flag_index}; expected address {byte_offset:#x} accessible, got length {len(prev)}"
        )
    if len(curr) <= byte_offset:
        raise ValueError(
            "curr buffer too small for event flag index "
            f"{flag_index}; expected address {byte_offset:#x} accessible, got length {len(curr)}"
        )

    bit = flag_index % 8
    mask = 1 << bit
    return not (prev[byte_offset] & mask) and (curr[byte_offset] & mask)


def check_goals(
    prev_mem: bytes,
    curr_mem: bytes,
    goals: Iterable[GoalDict],
) -> List[Tuple[str, float]]:
    """Check memory snapshots for goal completion.

    Parameters
    ----------
    prev_mem, curr_mem
        Consecutive WRAM snapshots as ``bytes`` or ``bytearray`` objects.
    goals
        Iterable of goal dictionaries. Each goal must contain ``id`` (str),
        ``type`` (``"map"`` or ``"event"``) and ``target_id`` (int). Optional
        ``reward`` values and ``prerequisites`` lists may be provided.

    Returns
    -------
    List[Tuple[str, float]]
        ``(goal_id, reward_value)`` pairs for each triggered goal.
    """

    triggered: List[Tuple[str, float]] = []

    map_changed, curr_map = _map_changed(prev_mem, curr_mem)

    for goal in goals:
        gtype = goal.get("type")
        target_id = int(goal.get("target_id", 0))
        reward = float(goal.get("reward", 1.0))

        if gtype == "map" and map_changed and curr_map == target_id:
            triggered.append((goal["id"], reward))
        elif gtype == "event":
            # First check badge flags
            if target_id < 8 and _badge_bit_set(prev_mem, curr_mem, target_id):
                triggered.append((goal["id"], reward))
            # Fallback to generic event flags if provided
            elif target_id >= 8 and _event_flag_set(prev_mem, curr_mem, target_id - 8):
                triggered.append((goal["id"], reward))

    return triggered
