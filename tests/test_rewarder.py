import pytest

from rewarder import predicate_from_goal, Rewarder
from poke_rewards import MAP_ID_ADDR, BADGE_FLAGS_ADDR


def make_mem(map_id: int = 0, badge_flags: int = 0, size: int = 0xE000) -> bytearray:
    mem = bytearray(size)
    mem[MAP_ID_ADDR] = map_id
    mem[BADGE_FLAGS_ADDR] = badge_flags
    return mem


class TestPredicateFromGoal:
    def test_map_goal_predicate(self):
        goal = {"id": "reach_city", "type": "map", "target_id": 1}
        pred = predicate_from_goal(goal)
        prev = make_mem(map_id=0)
        curr = make_mem(map_id=1)
        assert pred(prev, curr) is True
        # no change should be False
        assert not pred(curr, curr)

    def test_event_goal_predicate(self):
        goal = {"id": "defeat_brock", "type": "event", "target_id": 0}
        pred = predicate_from_goal(goal)
        prev = make_mem(badge_flags=0)
        curr = make_mem(badge_flags=0b00000001)
        assert pred(prev, curr) is True
        assert not pred(curr, curr)


class TestRewarderCompute:
    def test_compute_returns_sum_and_ids(self):
        goals = [
            {"id": "reach_city", "type": "map", "target_id": 1, "reward": 1.0},
            {"id": "defeat_brock", "type": "event", "target_id": 0, "reward": 5.0},
        ]
        rew = Rewarder(goals)
        prev = make_mem(map_id=0, badge_flags=0)
        curr = make_mem(map_id=1, badge_flags=0b00000001)
        total, triggered = rew.compute(prev, curr, env_reward=0.5)
        assert total == pytest.approx(6.5)
        assert set(triggered) == {"reach_city", "defeat_brock"}
