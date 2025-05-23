import unittest

try:
    import numpy as np
    import torch
    from ppo import ActorCritic, gather_rollout, Curriculum
except ModuleNotFoundError:  # PyTorch or numpy missing
    np = None
    torch = None


class DummyEnv:
    """Simple environment emitting observations in HWC format."""

    def __init__(self):
        self.observation_space = type("Space", (), {"shape": (4, 5, 3)})()
        self.action_space = type("Actions", (), {"n": 2})()
        self._step = 0

    def reset(self):
        self._step = 0
        return np.zeros(self.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        self._step += 1
        obs = np.full(self.observation_space.shape, self._step, dtype=np.uint8)
        return obs, 0.0, self._step >= 1, {}

    def get_ram(self):
        return bytearray(0xE000)


@unittest.skipUnless(np is not None and torch is not None, "PyTorch and numpy required")
class TestRolloutShapes(unittest.TestCase):
    def test_single_step_rollout(self):
        env = DummyEnv()
        obs_space_shape = env.observation_space.shape
        obs_shape = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1])
        model = ActorCritic(obs_shape, env.action_space.n)
        curriculum = Curriculum([])

        rollout = gather_rollout(env, model, curriculum, rollout_steps=1)
        self.assertEqual(len(rollout["states"]), 1)
        self.assertEqual(rollout["states"][0].shape, torch.Size(obs_shape))
        self.assertEqual(len(rollout["map_ids"]), 1)

    def test_obs_reset_after_done(self):
        """New observation from env.reset() should be used after episode end."""
        env = DummyEnv()
        obs_space_shape = env.observation_space.shape
        obs_shape = (obs_space_shape[2], obs_space_shape[0], obs_space_shape[1])
        model = ActorCritic(obs_shape, env.action_space.n)
        curriculum = Curriculum([])

        rollout = gather_rollout(env, model, curriculum, rollout_steps=2)
        self.assertEqual(len(rollout["states"]), 2)
        first_state = rollout["states"][0].numpy()
        second_state = rollout["states"][1].numpy()

        self.assertEqual(len(rollout["map_ids"]), 2)

        self.assertTrue(np.all(first_state == 0))
        self.assertTrue(np.all(second_state == 0))


if __name__ == "__main__":
    unittest.main()
