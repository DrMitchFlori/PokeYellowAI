import unittest

from rewarder import Rewarder
from poke_rewards import MAP_ID_ADDR, BADGE_FLAGS_ADDR


def make_mem(map_id: int = 0, badge_flags: int = 0, size: int = 0xE000) -> bytearray:
    mem = bytearray(size)
    mem[MAP_ID_ADDR] = map_id
    mem[BADGE_FLAGS_ADDR] = badge_flags
    return mem



class TestRewarderCompute(unittest.TestCase):
    def test_compute_returns_sum_and_ids(self):
        goals = [
            {"id": "reach_city", "type": "map", "target_id": 1, "reward": 1.0},
            {"id": "defeat_brock", "type": "event", "target_id": 0, "reward": 5.0},
        ]
        rew = Rewarder(goals)
        prev = make_mem(map_id=0, badge_flags=0)
        curr = make_mem(map_id=1, badge_flags=0b00000001)
        total, triggered = rew.compute(prev, curr, env_reward=0.5)
        self.assertAlmostEqual(total, 6.5)
        self.assertEqual(set(triggered), {"reach_city", "defeat_brock"})


if __name__ == "__main__":
    unittest.main()
