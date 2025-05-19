import unittest

from src.reward.predicates import MapPredicate, EventPredicate
from src.reward.rewarder import Rewarder
from poke_rewards import MAP_ID_ADDR, BADGE_FLAGS_ADDR, EVENT_FLAGS_BASE


def make_mem(map_id: int = 0, badge_flags: int = 0, event_flags: bytes | None = None, size: int = 0xE000) -> bytearray:
    mem = bytearray(size)
    mem[MAP_ID_ADDR] = map_id
    mem[BADGE_FLAGS_ADDR] = badge_flags
    if event_flags:
        for i, b in enumerate(event_flags):
            mem[EVENT_FLAGS_BASE + i] = b
    return mem


class TestPredicates(unittest.TestCase):
    def test_map_predicate(self):
        prev = make_mem(map_id=0)
        curr = make_mem(map_id=5)
        pred = MapPredicate(id="reach_map", reward=1.0, target_id=5)
        self.assertTrue(pred.triggered(prev, curr))

    def test_event_predicate_badge(self):
        prev = make_mem(badge_flags=0)
        curr = make_mem(badge_flags=0b00000010)
        pred = EventPredicate(id="badge", reward=2.0, target_id=1)
        self.assertTrue(pred.triggered(prev, curr))

    def test_event_predicate_flag(self):
        prev = make_mem(event_flags=bytes([0]))
        curr = make_mem(event_flags=bytes([1]))
        pred = EventPredicate(id="flag", reward=3.0, target_id=8)
        self.assertTrue(pred.triggered(prev, curr))

    def test_rewarder_accumulates(self):
        prev = make_mem(map_id=0, badge_flags=0)
        curr = make_mem(map_id=5, badge_flags=0b00000010)
        preds = [
            MapPredicate(id="reach_map", reward=1.0, target_id=5),
            EventPredicate(id="badge", reward=2.0, target_id=1),
        ]
        rewarder = Rewarder(preds)
        reward, triggered = rewarder.compute(prev, curr)
        self.assertEqual(reward, 3.0)
        self.assertEqual(set(triggered), {"reach_map", "badge"})
        self.assertEqual(rewarder.total_reward, 3.0)
