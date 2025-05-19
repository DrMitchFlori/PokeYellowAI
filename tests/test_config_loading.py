import os
import unittest

from ppo import load_config, yaml as yaml_lib

class TestConfigLoading(unittest.TestCase):
    def test_json_config(self):
        path = os.path.join(os.path.dirname(__file__), 'sample_config.json')
        cfg = load_config(path)
        self.assertEqual(cfg['ppo']['batch_size'], 32)
        self.assertEqual(cfg['curriculum']['threshold'], 0.5)

    @unittest.skipUnless(yaml_lib is not None, "PyYAML required")
    def test_yaml_config(self):
        path = os.path.join(os.path.dirname(__file__), 'sample_config.yaml')
        cfg = load_config(path)
        self.assertEqual(cfg['ppo']['batch_size'], 32)
        self.assertEqual(cfg['curriculum']['threshold'], 0.5)

if __name__ == '__main__':
    unittest.main()
