import unittest

from rewarder import Rewarder, predicate_from_goal
from poke_rewards import MAP_ID_ADDR, BADGE_FLAGS_ADDR


def make_mem(map_id: int = 0, badge_flags: int = 0, size: int = 0xE000) -> bytearray:
    mem = bytearray(size)
    mem[MAP_ID_ADDR] = map_id
    mem[BADGE_FLAGS_ADDR] = badge_flags
    return mem

class TestPredicateFromGoal(unittest.TestCase):
    def test_map_goal_predicate(self):
        goal = {
            "id": "reach_city",
            "type": "map",
            "target_id": 1,
            "reward": 1.0,
            "prerequisites": [],
        }
        pred = predicate_from_goal(goal)
        prev = make_mem(map_id=0)
        curr = make_mem(map_id=1)
        self.assertTrue(pred(prev, curr))
        # no change should be False
        self.assertFalse(pred(curr, curr))

    def test_event_goal_predicate(self):
        goal = {
            "id": "defeat_brock",
            "type": "event",
            "target_id": 0,
            "reward": 1.0,
            "prerequisites": [],
        }
        pred = predicate_from_goal(goal)
        prev = make_mem(badge_flags=0)
        curr = make_mem(badge_flags=0b00000001)
        self.assertTrue(pred(prev, curr))
        self.assertFalse(pred(curr, curr))

class TestRewarderCompute(unittest.TestCase):
    def test_compute_returns_sum_and_ids(self):
        goals = [
            {
                "id": "reach_city",
                "type": "map",
                "target_id": 1,
                "reward": 1.0,
                "prerequisites": [],
            },
            {
                "id": "defeat_brock",
                "type": "event",
                "target_id": 0,
                "reward": 5.0,
                "prerequisites": [],
            },
        ]
        rew = Rewarder(goals)
        prev = make_mem(map_id=0, badge_flags=0)
        curr = make_mem(map_id=1, badge_flags=0b00000001)
        total, triggered = rew.compute(prev, curr, env_reward=0.5)
        self.assertAlmostEqual(total, 6.5)
        self.assertEqual(set(triggered), {"reach_city", "defeat_brock"})


if __name__ == "__main__":
    unittest.main()
