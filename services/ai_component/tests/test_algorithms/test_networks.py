import unittest
import torch
from algorithms.networks import (
    CategoricalPolicy,
    GaussianPolicy,
    ValueNetwork,
    QNetwork,
    I2AImaginationNetwork
)

class TestNetworks(unittest.TestCase):
    def setUp(self):
        self.state_dim = 4
        self.action_dim = 2
        self.hidden_sizes = (256, 256)

    def test_categorical_policy(self):
        policy = CategoricalPolicy(self.state_dim, self.action_dim, self.hidden_sizes)
        state = torch.randn(1, self.state_dim)
        action_dist = policy(state)
        self.assertEqual(action_dist.probs.shape, (1, self.action_dim))

    def test_gaussian_policy(self):
        policy = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_sizes)
        state = torch.randn(1, self.state_dim)
        mean, log_std = policy(state)
        self.assertEqual(mean.shape, (1, self.action_dim))
        self.assertEqual(log_std.shape, (1, self.action_dim))

    def test_value_network(self):
        value_net = ValueNetwork(self.state_dim, self.hidden_sizes)
        state = torch.randn(1, self.state_dim)
        value = value_net(state)
        self.assertEqual(value.shape, (1, 1))

    def test_q_network(self):
        q_net = QNetwork(self.state_dim, self.action_dim, self.hidden_sizes)
        state = torch.randn(1, self.state_dim)
        action = torch.randn(1, self.action_dim)
        q1, q2 = q_net(state, action)
        self.assertEqual(q1.shape, (1, 1))
        self.assertEqual(q2.shape, (1, 1))

    def test_i2a_imagination_network(self):
        imagination_net = I2AImaginationNetwork(self.state_dim, self.action_dim, hidden_sizes=(512, 512, 512))
        initial_state = torch.randn(1, self.state_dim)
        action = torch.randn(1, self.action_dim)
        next_state = imagination_net(initial_state, action)
        self.assertEqual(next_state.shape, (1, self.state_dim))

    def test_weights_init(self):
        model = QNetwork(self.state_dim, self.action_dim)
        for param in model.parameters():
            if param.dim() == 2:  # weight
                self.assertTrue(torch.all(param.data != 0))  # Check weights are initialized
            elif param.dim() == 1:  # bias
                self.assertTrue(torch.all(param.data == 0))  # Check biases are initialized to 0

if __name__ == '__main__':
    unittest.main()
