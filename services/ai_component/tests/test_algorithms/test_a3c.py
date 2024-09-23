import unittest
import torch
import gymnasium as gym
from algorithms.a3c import A3C, A3CWorker
from algorithms.networks import ActorCritic

class TestA3C(unittest.TestCase):
    def setUp(self):
        self.config = {
            'policy_lr': 0.001,
            'max_episodes': 10,
            'num_workers': 2,
            'device': 'cpu',
            'gamma': 0.99,
            'tau': 0.95,
            'max_episode_length': 200
        }
        self.env = gym.make('CartPole-v1')
        self.a3c = A3C(self.config, self.env)

    def test_setup(self):
        self.a3c.setup()
        self.assertIsNotNone(self.a3c.global_network)
        self.assertIsNotNone(self.a3c.optimizer)

    def test_select_action(self):
        self.a3c.setup()
        state = self.env.reset()[0]
        action = self.a3c.select_action(state)
        self.assertIn(action, range(self.env.action_space.n))

    def test_train_step(self):
        self.a3c.setup()
        self.a3c.train_step()  # This should run without errors

    def test_save_load(self):
        self.a3c.setup()
        filepath = 'test_a3c.pth'
        self.a3c.save(filepath)
        self.a3c.load(filepath)
        self.assertIsNotNone(self.a3c.global_network.state_dict())
        self.assertIsNotNone(self.a3c.optimizer.state_dict())

    def test_a3c_worker(self):
        global_network = ActorCritic(self.env.observation_space.shape[0], self.env.action_space.n)
        optimizer = torch.optim.Adam(global_network.parameters(), lr=self.config['policy_lr'])
        worker = A3CWorker(global_network, optimizer, self.a3c.global_episode, self.config, 0, self.env)
        worker.start()
        worker.join()  # Ensure the worker runs without errors

if __name__ == '__main__':
    unittest.main()
