import json
from typing import Any, Dict, List

import retro

# WRAM addresses used for tracking progress. These are specific to
# Pokemon Yellow and may need to be adjusted for other ROMs.
WRAM_ADDRESSES = [0xD35D, 0xD355, 0xD754, 0xD75D]


def connect_emulator(game_name: str, state: str | None = None) -> retro.RetroEnv:
    """Return a Gym Retro environment for the given game name.

    Parameters
    ----------
    game_name: str
        The identifier of the Gym Retro game. The ROM must be installed and
        registered with Gym Retro.
    state: str | None, optional
        Optional saved state to load when starting the environment.
    """
    env = retro.make(game=game_name, state=state)
    return env


def read_wram_bytes(env: retro.RetroEnv) -> Dict[int, int]:
    """Read bytes from the emulator's WRAM at predefined addresses."""
    ram_snapshot = {}
    for address in WRAM_ADDRESSES:
        value = env.get_memory(address, 1)[0]
        ram_snapshot[address] = value
    return ram_snapshot


def load_goal_spec(path: str) -> Dict[str, Any]:
    """Load a goal specification from a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def evaluate_goals(ram_snapshot: Dict[int, int], spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Evaluate goal predicates defined in the specification.

    Parameters
    ----------
    ram_snapshot: Dict[int, int]
        Mapping of WRAM addresses to the current byte value at that address.
    spec: Dict[str, Any]
        Goal specification loaded from ``load_goal_spec``.

    Returns
    -------
    List[Dict[str, Any]]
        A list of dictionaries describing whether each goal has been met.
    """
    results = []
    for goal in spec.get("goals", []):
        predicates = goal.get("predicates", [])
        met = True
        for pred in predicates:
            address = int(pred["address"], 16)
            expected = int(pred["equals"], 0)
            if ram_snapshot.get(address) != expected:
                met = False
                break
        results.append({"description": goal.get("description", ""), "met": met})
    return results
