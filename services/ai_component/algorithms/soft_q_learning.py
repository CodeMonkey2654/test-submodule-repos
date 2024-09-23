import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.networks import SoftQNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
from torch.distributions import Normal
import gymnasium as gym


class SoftQLearning(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        self.q_network = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network = SoftQNetwork(state_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config['q_lr'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)  # For soft updates
        self.alpha = self.config.get('alpha', 0.2)
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            # Sample action from a Gaussian policy
            mean = torch.zeros(self.env.action_space.shape[0]).to(self.device)
            std = torch.ones(self.env.action_space.shape[0]).to(self.device)
            dist = Normal(mean, std)
            action = dist.sample().clamp(-1, 1) if not evaluate else torch.tanh(mean)
        return action.cpu().numpy()

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return  # Not enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Current Q estimates
        current_q = self.q_network(states, actions)

        # Next actions and log probabilities
        with torch.no_grad():
            mean = torch.zeros_like(next_states)
            std = torch.ones_like(next_states)
            dist = Normal(mean, std)
            next_actions = dist.sample().clamp(-1, 1)
            log_prob = dist.log_prob(next_actions).sum(dim=-1, keepdim=True)
            target_q = self.target_q_network(next_states, next_actions) - self.alpha * log_prob
            target_q = rewards + (1 - dones) * self.gamma * target_q

        # Loss
        loss = F.mse_loss(current_q, target_q)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_q_network.parameters(), self.q_network.parameters()):
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
