"""Combine goal predicates and produce shaped rewards."""

from typing import Iterable, List, Tuple

from .predicates import Predicate


class Rewarder:
    """Evaluate a set of predicates and accumulate rewards."""

    def __init__(self, predicates: Iterable[Predicate]):
        self.predicates = list(predicates)
        self.total_reward = 0.0

    def compute(self, prev: bytes, curr: bytes) -> Tuple[float, List[str]]:
        """Return additional reward and triggered predicate ids."""
        triggered: List[str] = []
        reward = 0.0
        for p in self.predicates:
            if p.triggered(prev, curr):
                triggered.append(p.id)
                reward += p.reward
        self.total_reward += reward
        return reward, triggered

    def reset_total(self) -> None:
        self.total_reward = 0.0
