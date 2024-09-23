import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.networks import QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import numpy as np
import gymnasium as gym

class SoftQLearning(BaseAlgorithm):
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
        self.tau = self.config.get('tau', 0.005)
        self.alpha = self.config.get('alpha', 0.2)  # Temperature parameter
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state, torch.zeros(1, self.action_dim).to(self.device))[0]
            if self.discrete:
                if evaluate:
                    return q_values.argmax().item()
                else:
                    probs = F.softmax(q_values / self.alpha, dim=-1)
                    return np.random.choice(self.action_dim, p=probs.cpu().numpy())
            else:
                if evaluate:
                    return q_values.cpu().numpy()
                else:
                    return np.random.normal(q_values.cpu().numpy(), self.alpha)

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_q1, next_q2 = self.target_q_network(next_state_batch, torch.zeros_like(action_batch))
            next_q = torch.min(next_q1, next_q2)
            if self.discrete:
                next_v = self.alpha * torch.log(torch.sum(torch.exp(next_q / self.alpha), dim=1, keepdim=True))
            else:
                next_v = next_q - self.alpha * torch.log(2 * np.pi * self.alpha) - 0.5
            target_q = reward_batch + (1 - done_batch) * self.gamma * next_v

        current_q1, current_q2 = self.q_network(state_batch, action_batch)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        self.optimizer.zero_grad()
        q_loss.backward()
        self.optimizer.step()

        # Soft update of the target network
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
