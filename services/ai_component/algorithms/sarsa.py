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
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['q_lr'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=False)

        self.gamma = self.config.get('gamma', 0.99)
        self.batch_size = self.config.get('batch_size', 256)
        self.noise_std = self.config.get('noise_std', 0.1)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)
            action = q_values.cpu().numpy()
            if not evaluate:
                action += np.random.normal(0, self.noise_std, size=action.shape)
                action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return  # Not enough samples

        states, actions, rewards, next_states, next_actions, dones = self.replay_buffer.sample_sarsa(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        next_actions = torch.FloatTensor(next_actions).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q values
        current_q = self.q_network(states)

        # Next Q values
        next_q = self.q_network(next_states)

        # Target Q values
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
