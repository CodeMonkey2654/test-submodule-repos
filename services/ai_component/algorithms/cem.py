import torch
import torch.optim as optim
from algorithms.networks import BasePolicyNetwork
from algorithms.base import BaseAlgorithm
import numpy as np
import gymnasium as gym


class CEM(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        self.policy = BasePolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config['policy_lr'])

        self.num_iterations = self.config.get('num_iterations', 10)
        self.population_size = self.config.get('population_size', 50)
        self.top_percent = self.config.get('top_percent', 0.2)
        self.sample_episodes = self.config.get('sample_episodes', 5)
        self.elite_size = max(1, int(self.population_size * self.top_percent))

        # Initialize mean and std for parameter perturbations
        self.param_mean = self.get_flat_params().clone()
        self.param_std = torch.ones_like(self.param_mean) * self.config.get('initial_std', 0.1)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(state)
            if evaluate:
                action = mean.cpu().numpy()[0]
            else:
                action = torch.normal(mean, std).cpu().numpy()[0]
        return action

    def train_step(self):
        # Sample a population of perturbations
        noise = torch.randn(self.population_size, self.param_mean.shape[0]).to(self.device) * self.param_std
        perturbed_params = self.param_mean.unsqueeze(0) + noise  # Shape: (population_size, num_params)

        rewards = []

        for i in range(self.population_size):
            # Apply perturbation to the policy
            self.set_flat_params(perturbed_params[i])

            # Evaluate the perturbed policy
            total_reward = self.evaluate_policy(self.sample_episodes)
            rewards.append(total_reward)

        rewards = np.array(rewards)

        # Select elite perturbations
        elite_indices = rewards.argsort()[-self.elite_size:]
        elite_params = perturbed_params[elite_indices]

        # Update mean and std based on elite perturbations
        elite_params_np = elite_params.cpu().numpy()
        self.param_mean = torch.from_numpy(np.mean(elite_params_np, axis=0)).to(self.device)
        self.param_std = torch.from_numpy(np.std(elite_params_np, axis=0)).to(self.device) + 1e-6  # Add small value to prevent std=0

        # Update the policy with the new mean parameters
        self.set_flat_params(self.param_mean)

    def evaluate_policy(self, episodes=5):
        """
        Evaluate the current policy over a number of episodes and return the average total reward.
        """
        total_rewards = []
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.select_action(state, evaluate=True)
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            total_rewards.append(episode_reward)
        average_reward = np.mean(total_rewards)
        return average_reward

    def get_flat_params(self):
        """
        Get the policy's parameters as a flat tensor.
        """
        return torch.cat([param.view(-1) for param in self.policy.parameters()])

    def set_flat_params(self, flat_params):
        """
        Set the policy's parameters from a flat tensor.
        """
        pointer = 0
        for param in self.policy.parameters():
            num_params = param.numel()
            param.data.copy_(flat_params[pointer:pointer + num_params].view_as(param))
            pointer += num_params

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'param_mean': self.param_mean.cpu(),
            'param_std': self.param_std.cpu(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.param_mean = checkpoint['param_mean'].to(self.device)
        self.param_std = checkpoint['param_std'].to(self.device)
        self.set_flat_params(self.param_mean)
