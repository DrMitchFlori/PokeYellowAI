import unittest

from ppo import Curriculum


class TestCurriculumUnlocks(unittest.TestCase):
    def test_goal_unlock_at_threshold(self):
        goals = [
            {"id": "first", "type": "event", "target_id": 0},
            {"id": "second", "type": "event", "target_id": 1, "prerequisites": ["first"]},
        ]
        curriculum = Curriculum(goals, threshold=0.75)

        # Only the first goal should be active initially
        active_ids = {g["id"] for g in curriculum.active_goals()}
        self.assertEqual(active_ids, {"first"})

        # Episode 1: fail the first goal
        curriculum.record_episode([])
        active_ids = {g["id"] for g in curriculum.active_goals()}
        self.assertEqual(active_ids, {"first"})

        # Episodes 2 and 3: succeed at the first goal
        curriculum.record_episode(["first"])
        curriculum.record_episode(["first"])
        active_ids = {g["id"] for g in curriculum.active_goals()}
        self.assertNotIn("second", active_ids)

        # Episode 4: third success brings success rate to 3/4 = 0.75
        curriculum.record_episode(["first"])
        active_ids = {g["id"] for g in curriculum.active_goals()}
        self.assertIn("second", active_ids)


if __name__ == "__main__":
    unittest.main()
