import unittest

from poke_rewards import (
    check_goals,
    MAP_ID_ADDR,
    BADGE_FLAGS_ADDR,
    EVENT_FLAGS_BASE,
    _map_changed,
    _badge_bit_set,
    _event_flag_set,
)


def make_mem(map_id: int = 0, badge_flags: int = 0, size: int = 0xE000) -> bytearray:
    mem = bytearray(size)
    mem[MAP_ID_ADDR] = map_id
    mem[BADGE_FLAGS_ADDR] = badge_flags
    return mem


class TestPokeRewards(unittest.TestCase):
    def test_map_id_address_constant(self):
        """Ensure MAP_ID_ADDR matches the documented location."""
        self.assertEqual(MAP_ID_ADDR, 0xD35D)

    def test_map_transition_triggers_goal(self):
        prev = make_mem(map_id=0)
        curr = make_mem(map_id=1)

        goals = [
            {"id": "reach_viridian_city", "type": "map", "target_id": 1, "reward": 1.0}
        ]

        triggered = check_goals(prev, curr, goals)
        self.assertEqual(triggered, [("reach_viridian_city", 1.0)])

    def test_event_flag_triggers_goal(self):
        prev = make_mem(badge_flags=0b00000000)
        curr = make_mem(badge_flags=0b00000001)

        goals = [
            {"id": "defeat_brock", "type": "event", "target_id": 0, "reward": 5.0}
        ]

        triggered = check_goals(prev, curr, goals)
        self.assertEqual(triggered, [("defeat_brock", 5.0)])

    def test_no_trigger_when_values_unchanged(self):
        prev = make_mem(map_id=1, badge_flags=0b00000001)
        curr = make_mem(map_id=1, badge_flags=0b00000001)

        goals = [
            {"id": "reach_viridian_city", "type": "map", "target_id": 1, "reward": 1.0},
            {"id": "defeat_brock", "type": "event", "target_id": 0, "reward": 5.0},
        ]

        triggered = check_goals(prev, curr, goals)
        self.assertEqual(triggered, [])

    def test_multiple_goals_same_frame(self):
        prev = make_mem(map_id=0, badge_flags=0b00000000)
        curr = make_mem(map_id=1, badge_flags=0b00000001)

        goals = [
            {"id": "reach_viridian_city", "type": "map", "target_id": 1, "reward": 1.0},
            {"id": "defeat_brock", "type": "event", "target_id": 0, "reward": 5.0},
        ]

        triggered = check_goals(prev, curr, goals)
        self.assertEqual(sorted(triggered), [
            ("defeat_brock", 5.0),
            ("reach_viridian_city", 1.0),
        ])

    def test_map_changed_short_buffer(self):
        prev = bytearray(MAP_ID_ADDR)
        curr = bytearray(MAP_ID_ADDR + 1)
        with self.assertRaises(ValueError) as ctx:
            _map_changed(prev, curr)
        self.assertIn("too small", str(ctx.exception))

    def test_badge_bit_set_short_buffer(self):
        prev = bytearray(BADGE_FLAGS_ADDR)
        curr = bytearray(BADGE_FLAGS_ADDR)
        with self.assertRaises(ValueError) as ctx:
            _badge_bit_set(prev, curr, 0)
        self.assertIn("too small", str(ctx.exception))

    def test_event_flag_set_short_buffer(self):
        flag_index = 0
        offset = EVENT_FLAGS_BASE + flag_index // 8
        prev = bytearray(offset)
        curr = bytearray(offset)
        with self.assertRaises(ValueError) as ctx:
            _event_flag_set(prev, curr, flag_index)
        self.assertIn("too small", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
