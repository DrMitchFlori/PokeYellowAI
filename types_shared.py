"""Common type aliases for PokeYellowAI modules."""

from typing import List, TypedDict


class GoalDict(TypedDict):
    """Structured representation of a training goal."""

    id: str
    type: str
    target_id: int
    reward: float
    prerequisites: List[str]

