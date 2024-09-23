import unittest
import torch
import gymnasium as gym
from algorithms.a2c import A2C

class TestA2C(unittest.TestCase):
    def setUp(self):
        config = {
            'policy_lr': 0.001,
            'buffer_size': 1000,
            'gamma': 0.99,
            'tau': 0.95,
            'batch_size': 32,
            'entropy_coeff': 0.001
        }
        env = gym.make('CartPole-v1')
        self.a2c = A2C(config, env)

    def test_setup(self):
        self.assertIsNotNone(self.a2c.actor_critic)
        self.assertIsNotNone(self.a2c.optimizer)
        self.assertIsNotNone(self.a2c.rollout_buffer)
        self.assertEqual(self.a2c.gamma, 0.99)
        self.assertEqual(self.a2c.tau, 0.95)
        self.assertEqual(self.a2c.batch_size, 32)

    def test_select_action_discrete(self):
        state = self.a2c.env.reset()[0]
        action, log_prob = self.a2c.select_action(state)
        self.assertIsInstance(action, int)
        self.assertIsInstance(log_prob, float)

    def test_select_action_continuous(self):
        self.a2c.env = gym.make('Pendulum-v1')
        self.a2c.setup()  # Re-setup for the new environment
        state = self.a2c.env.reset()[0]
        action, log_prob = self.a2c.select_action(state)
        self.assertIsInstance(action, torch.Tensor)
        self.assertEqual(action.shape[0], self.a2c.env.action_space.shape[0])

    def test_compute_gae(self):
        rewards = [1.0, 1.0, 1.0]
        values = [0.5, 0.5, 0.5]
        next_values = [0.6, 0.6, 0.6]
        dones = [0, 0, 1]
        advantages = self.a2c.compute_gae(rewards, values, next_values, dones)
        self.assertEqual(len(advantages), len(rewards))

    def test_train_step(self):
        for _ in range(10):  # Fill the buffer with some data
            state = self.a2c.env.reset()[0]
            action = self.a2c.env.action_space.sample()
            self.a2c.rollout_buffer.add(state, action, 1.0, state, False, 0.0)  # Dummy values

        try:
            self.a2c.train_step()  # Should not raise any exceptions
        except Exception as e:
            self.fail(f"train_step() raised {type(e).__name__} unexpectedly!")

    def test_save_load(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdirname:
            filepath = os.path.join(tmpdirname, "a2c_test.pth")
            self.a2c.save(filepath)
            new_a2c = A2C(self.a2c.config, self.a2c.env)
            new_a2c.load(filepath)
            
            self.assertEqual(
                self.a2c.actor_critic.state_dict().keys(),
                new_a2c.actor_critic.state_dict().keys()
            )
            self.assertEqual(
                self.a2c.optimizer.state_dict().keys(),
                new_a2c.optimizer.state_dict().keys()
            )

if __name__ == '__main__':
    unittest.main()
