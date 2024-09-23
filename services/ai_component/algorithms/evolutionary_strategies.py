import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import GaussianPolicy, CategoricalPolicy
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class EvolutionaryStrategies(BaseAlgorithm):
    def setup(self):
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            self.action_dim = self.env.action_space.n
            self.discrete = True
            self.policy = CategoricalPolicy(self.state_dim, self.action_dim).to(self.device)
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.discrete = False
            self.policy = GaussianPolicy(self.state_dim, self.action_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['learning_rate'])

        self.population_size = self.config.get('population_size', 50)
        self.noise_std = self.config.get('noise_std', 0.1)
        self.learning_rate = self.config.get('learning_rate', 0.01)
        self.num_iterations = self.config.get('num_iterations', 100)
        self.elite_frac = self.config.get('elite_frac', 0.2)
        self.sigma = self.config.get('sigma', 0.1)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.discrete:
                dist = self.policy(state)
                if evaluate:
                    action = dist.probs.argmax().item()
                else:
                    action = dist.sample().item()
            else:
                mean, log_std = self.policy(state)
                if evaluate:
                    action = mean.cpu().numpy()[0]
                else:
                    action, _, _ = self.policy.sample(state)
                    action = action.cpu().numpy()[0]
        return action

    def train_step(self):
        # Generate population
        theta = torch.cat([p.data.view(-1) for p in self.policy.parameters()])
        population = [theta + self.sigma * torch.randn_like(theta) for _ in range(self.population_size)]

        # Evaluate population
        rewards = []
        for params in population:
            self.set_params(params)
            reward = self.evaluate_policy()
            rewards.append(reward)

        # Compute ranks and elite
        rewards = np.array(rewards)
        ranks = np.argsort(rewards)[::-1]
        elite_size = int(self.population_size * self.elite_frac)
        elite = [population[i] for i in ranks[:elite_size]]

        # Update policy
        theta = torch.mean(torch.stack(elite), dim=0)
        self.set_params(theta)

    def set_params(self, params):
        idx = 0
        for p in self.policy.parameters():
            size = p.numel()
            p.data.copy_(params[idx:idx + size].view_as(p))
            idx += size

    def evaluate_policy(self, episodes=5):
        total_reward = 0.0
        for _ in range(episodes):
            state, _ = self.env.reset()
            done = False
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
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
