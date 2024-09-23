import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import Actor, Critic
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym


class PPO(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim).to(self.device)

        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()),
                                    lr=self.config['learning_rate'])

        self.buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)
        self.eps_clip = self.config.get('eps_clip', 0.2)
        self.K_epochs = self.config.get('K_epochs', 4)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        mean, std = self.actor(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(dim=-1)
        return action.cpu().detach().numpy(), action_logprob.cpu().detach().numpy()

    def train_step(self):
        # Convert buffer to tensors
        states = torch.FloatTensor(self.buffer.states).to(self.device)
        actions = torch.FloatTensor(self.buffer.actions).to(self.device)
        rewards = torch.FloatTensor(self.buffer.rewards).to(self.device)
        dones = torch.FloatTensor(self.buffer.dones).to(self.device)
        old_logprobs = torch.FloatTensor(self.buffer.logprobs).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, states)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            logprobs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            state_values = self.critic(states).squeeze()

            ratios = torch.exp(logprobs - old_logprobs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = torch.nn.functional.mse_loss(state_values, returns)

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Clear buffer
        self.buffer.clear()

    def compute_gae(self, rewards, dones, states):
        with torch.no_grad():
            state_values = self.critic(states).squeeze()
            next_values = torch.cat((state_values[1:], torch.zeros(1).to(self.device)))
            deltas = rewards + self.gamma * next_values * (1 - dones) - state_values
            advantages = torch.zeros_like(rewards).to(self.device)
            gae = 0
            for step in reversed(range(len(rewards))):
                gae = deltas[step] + self.gamma * self.lam * (1 - dones[step]) * gae
                advantages[step] = gae
            returns = advantages + state_values
        return advantages, returns

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
