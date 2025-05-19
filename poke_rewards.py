"""Reward detection utilities for Pokémon Yellow WRAM snapshots."""

from typing import Dict, Iterable, List, Tuple

# Memory addresses based on community documentation
MAP_ID_ADDR = 0xD35E
BADGE_FLAGS_ADDR = 0xD356
EVENT_FLAGS_BASE = 0xD747


Goal = Dict[str, object]


def _map_changed(prev: bytes, curr: bytes) -> Tuple[bool, int]:
    """Return (changed, current map ID)."""
    prev_id = prev[MAP_ID_ADDR]
    curr_id = curr[MAP_ID_ADDR]
    return (prev_id != curr_id, curr_id)


def _badge_bit_set(prev: bytes, curr: bytes, bit: int) -> bool:
    """Return True if the given badge bit transitioned from 0 to 1."""
    mask = 1 << bit
    return not (prev[BADGE_FLAGS_ADDR] & mask) and (curr[BADGE_FLAGS_ADDR] & mask)


def _event_flag_set(prev: bytes, curr: bytes, flag_index: int) -> bool:
    """Return True if a generic event flag bit transitioned from 0 to 1."""
    byte_offset = EVENT_FLAGS_BASE + flag_index // 8
    bit = flag_index % 8
    mask = 1 << bit
    return not (prev[byte_offset] & mask) and (curr[byte_offset] & mask)


def check_goals(
    prev_mem: bytes,
    curr_mem: bytes,
    goals: Iterable[Goal],
    map_ids: Dict[str, int],
    flags: Dict[str, int],
) -> List[Tuple[str, float]]:
    """Check memory snapshots for goal completion.

    Parameters
    ----------
    prev_mem, curr_mem
        Consecutive WRAM snapshots as ``bytes`` or ``bytearray`` objects.
    goals
        Iterable of goal dictionaries.  Each goal must contain ``id`` (str),
        ``type`` (``"map"`` or ``"event"``), and ``target_id`` (int).  Goals may
        optionally include ``reward`` (float).
    map_ids
        Mapping of map names to their numeric identifiers.  Currently used only
        for reference.
    flags
        Mapping of event flag descriptions to their memory offsets.  Currently
        used only when ``type`` is ``"event"`` and the target refers to a generic
        flag rather than badge flags.

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
