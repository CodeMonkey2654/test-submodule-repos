import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from algorithms.networks import QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class DQN(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.discrete = True
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.discrete = False

        hidden_sizes = self.config.get('hidden_sizes', (256, 256, 256))
        self.q_network = QNetwork(state_dim, self.action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_q_network = QNetwork(state_dim, self.action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['learning_rate'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, self.action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.batch_size = self.config.get('batch_size', 64)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.update_target_every = self.config.get('update_target_every', 100)
        self.train_steps = 0

    def select_action(self, state, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            if self.discrete:
                return np.random.randint(self.action_dim)
            else:
                return np.random.uniform(self.env.action_space.low, self.env.action_space.high)
        
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values, _ = self.q_network(state, None)
            if self.discrete:
                return q_values.argmax(dim=1).item()
            else:
                return q_values.cpu().numpy().squeeze()

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q1, current_q2 = self.q_network(states, actions)
        current_q_values = torch.min(current_q1, current_q2)

        with torch.no_grad():
            next_q1, next_q2 = self.target_q_network(next_states, None)
            next_q_values = torch.min(next_q1, next_q2).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = F.mse_loss(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.update_target_every == 0:
            self.soft_update_target_network()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def soft_update_target_network(self):
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_steps = checkpoint['train_steps']
