import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.networks import BasePolicyNetwork
from algorithms.base import BaseAlgorithm
import numpy as np
import gymnasium as gym


class EvolutionStrategies(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        self.policy = BasePolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['policy_lr'])

        self.population_size = self.config.get('population_size', 50)
        self.noise_std = self.config.get('noise_std', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.03)
        self.num_iterations = self.config.get('num_iterations', 100)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.policy(state).cpu().numpy()[0]
        return action

    def train_step(self):
        # Sample noise
        noise = torch.randn(self.population_size, self.policy.num_parameters()).to(self.device) * self.noise_std

        # Evaluate each candidate
        rewards = []
        for i in range(self.population_size):
            self.policy.apply_noise(noise[i])
            total_reward = self.evaluate_policy()
            rewards.append(total_reward)
            self.policy.remove_noise()

        rewards = np.array(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Update policy
        gradients = torch.zeros_like(self.policy.parameters()).to(self.device)
        for i in range(self.population_size):
            self.policy.apply_noise(noise[i])
            gradients += rewards[i] * noise[i]
            self.policy.remove_noise()
        gradients /= self.population_size * self.noise_std

        # Apply gradient
        for param, grad in zip(self.policy.parameters(), gradients):
            param.grad = grad
        self.optimizer.step()

    def evaluate_policy(self, episodes=1):
        total_reward = 0.0
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, done, _ = self.env.step(action)
                total_reward += reward
        return total_reward / episodes

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
