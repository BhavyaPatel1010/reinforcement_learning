import unittest
import numpy as np
from src.inventory_env import InventoryEnv


class EvaluationTests(unittest.TestCase):
    def test_test_demand_does_not_change_action_or_observation_scales(self):
        train = InventoryEnv([10] * 20, [2] * 20)
        test = InventoryEnv([100] * 20, [20] * 20, reference_stats=train.reference_stats)
        np.testing.assert_array_equal(train.action_values, test.action_values)
        self.assertEqual(train.max_inventory, test.max_inventory)
        self.assertEqual(train.reward_scale, test.reward_scale)
        self.assertEqual(test.reset()[6], 10)

    def test_pipeline_distinguishes_delivery_dates(self):
        env = InventoryEnv([10] * 20)
        env.reset()
        env.step(5)
        self.assertEqual(len(env._state()), env.state_size)
        self.assertEqual(env._state()[-2], 0)
        self.assertGreater(env._state()[-1], 0)
        env.step(0)
        self.assertGreater(env._state()[-2], 0)
        self.assertEqual(env._state()[-1], 0)


if __name__ == '__main__':
    unittest.main()
