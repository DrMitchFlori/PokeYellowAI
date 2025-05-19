"""Common type aliases for PokeYellowAI modules."""

from typing import List, TypedDict

try:  # Python >=3.11
    from typing import NotRequired
except ImportError:  # pragma: no cover - fallback for older versions
    from typing_extensions import NotRequired


class GoalDict(TypedDict):
    """Structured representation of a training goal."""

    id: str
    type: str
    target_id: int
    reward: NotRequired[float]
    prerequisites: NotRequired[List[str]]

