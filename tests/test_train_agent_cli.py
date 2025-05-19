import unittest


class TestTrainAgentCLI(unittest.TestCase):
    def test_output_model_argument_present(self):
        with open('train_agent.py', 'r', encoding='utf-8') as f:
            src = f.read()
        self.assertIn('--output-model', src)
        self.assertIn('args.output_model', src)


if __name__ == '__main__':
    unittest.main()
