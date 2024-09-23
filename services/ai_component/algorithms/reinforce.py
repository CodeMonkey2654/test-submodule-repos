import torch
import torch.optim as optim
from algorithms.networks import CategoricalPolicy
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class REINFORCE(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            self.discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            self.discrete = False

        hidden_sizes = self.config.get('hidden_sizes', (256, 256))
        self.policy = CategoricalPolicy(state_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=self.discrete)

        self.gamma = self.config.get('gamma', 0.99)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            dist = self.policy(state)
            if evaluate:
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.sample()
        return action.cpu().numpy().squeeze()

    def train_step(self):
        states, actions, rewards, _, dones, _ = self.rollout_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)

        # Calculate returns
        returns = []
        G = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            G = reward + self.gamma * G * (1 - done)
            returns.insert(0, G)
        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Calculate loss
        dist = self.policy(states)
        log_probs = dist.log_prob(actions)
        loss = -(log_probs * returns).mean()

        # Optimize the policy
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear the rollout buffer
        self.rollout_buffer.clear()

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
