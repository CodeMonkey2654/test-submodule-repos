import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from algorithms.networks import PolicyNetwork
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class REINFORCE(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            discrete = False

        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['policy_lr'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        logits = self.policy(state)
        if evaluate:
            action = torch.argmax(logits).item()
        else:
            dist = F.softmax(logits, dim=-1)
            dist = Categorical(dist)
            action = dist.sample().item()
        return action

    def train_step(self):
        if self.rollout_buffer.size < self.batch_size:
            return  # Not enough samples

        states, actions, rewards, _, _, _, returns = self.rollout_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        logits = self.policy(states)
        dist = F.softmax(logits, dim=-1)
        dist = Categorical(dist)

        log_probs = dist.log_prob(actions)
        loss = -(log_probs * returns).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
