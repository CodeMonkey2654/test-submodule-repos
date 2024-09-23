import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.networks import QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import numpy as np
import gymnasium as gym

class SARSA(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.discrete = False

        hidden_sizes = self.config.get('hidden_sizes', (256, 256))
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_sizes).to(self.device)
        self.target_q_network = QNetwork(state_dim, self.action_dim, hidden_sizes).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, self.action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.batch_size = self.config.get('batch_size', 64)
        self.tau = self.config.get('tau', 0.005)
        self.epsilon = self.config.get('epsilon', 0.1)

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.random() < self.epsilon:
            if self.discrete:
                return np.random.randint(self.action_dim)
            else:
                return np.random.uniform(self.env.action_space.low, self.env.action_space.high)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state, torch.zeros(1, self.action_dim).to(self.device))[0]
            if self.discrete:
                return q_values.argmax().item()
            else:
                return q_values.cpu().numpy()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Get current Q values
        current_q_values = self.q_network(state_batch, action_batch)[0]

        # Get next action using current policy (SARSA is on-policy)
        with torch.no_grad():
            next_actions = self.select_action(next_state_batch.cpu().numpy())
            next_actions = torch.FloatTensor(next_actions).to(self.device)

            # Get next Q values from target network
            next_q_values = self.target_q_network(next_state_batch, next_actions)[0]

        # Compute target Q values
        target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        for param, target_param in zip(self.q_network.parameters(), self.target_q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
