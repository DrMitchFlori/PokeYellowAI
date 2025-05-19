import unittest
import types

import ppo
from ppo import compute_gae


class FakeTensor:
    def __init__(self, data):
        self.data = list(data)

    def __add__(self, other):
        if isinstance(other, FakeTensor):
            return FakeTensor([a + b for a, b in zip(self.data, other.data)])
        return FakeTensor([a + other for a in self.data])

    def __iter__(self):
        return iter(self.data)

    def __repr__(self):
        return f"FakeTensor({self.data!r})"


class FakeTorch(types.SimpleNamespace):
    float32 = "float32"

    @staticmethod
    def tensor(data, dtype=None):
        return FakeTensor(data)


def approx_equal(seq1, seq2, tol=1e-6):
    return all(abs(a - b) <= tol for a, b in zip(seq1, seq2))


class TestComputeGAE(unittest.TestCase):
    def test_short_sequence(self):
        rewards = [1.0, 2.0, 3.0]
        values = [0.5, 1.0, 1.5]
        dones = [False, False, True]
        gamma = 0.99
        lam = 0.95

        old_torch = ppo.torch
        ppo.torch = FakeTorch()
        try:
            advantages, returns = compute_gae(rewards, values, dones, gamma=gamma, lam=lam)
        finally:
            ppo.torch = old_torch

        # Compute expected results manually
        adv = 0.0
        expected_adv = []
        last_value = 0.0
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            delta = r + gamma * last_value * (1.0 - float(d)) - v
            adv = delta + gamma * lam * (1.0 - float(d)) * adv
            expected_adv.insert(0, adv)
            last_value = v
        expected_ret = [a + v for a, v in zip(expected_adv, values)]

        self.assertTrue(approx_equal(advantages.data, expected_adv))
        self.assertTrue(approx_equal(returns.data, expected_ret))


if __name__ == "__main__":
    unittest.main()
