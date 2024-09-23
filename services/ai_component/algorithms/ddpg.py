import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import GaussianPolicy, QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class DDPG(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            raise ValueError("DDPG is designed for continuous action spaces.")
        action_dim = self.env.action_space.shape[0]
        
        self.action_limit = float(self.env.action_space.high[0])

        self.actor = GaussianPolicy(state_dim, action_dim, hidden_sizes=(256, 256), action_limit=self.action_limit).to(self.device)
        self.actor_target = GaussianPolicy(state_dim, action_dim, hidden_sizes=(256, 256), action_limit=self.action_limit).to(self.device)
        self.critic = QNetwork(state_dim, action_dim, hidden_sizes=(256, 256)).to(self.device)
        self.critic_target = QNetwork(state_dim, action_dim, hidden_sizes=(256, 256)).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.config['actor_lr'])
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.config['critic_lr'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.noise_std = self.config.get('noise_std', 0.1)
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _, _ = self.actor.sample(state)
            action = action.cpu().numpy().squeeze()
        if not evaluate:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -self.action_limit, self.action_limit)
        return action

    def train_step(self):
        if self.replay_buffer.size < self.batch_size:
            return

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Compute the target Q value
        with torch.no_grad():
            next_action, _, _ = self.actor_target.sample(next_state_batch)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward_batch + (1.0 - done_batch) * self.gamma * target_Q

        # Get current Q estimate
        current_Q1, current_Q2 = self.critic(state_batch, action_batch)

        # Compute critic loss
        critic_loss = torch.nn.functional.mse_loss(current_Q1, target_Q) + torch.nn.functional.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_action, _, _ = self.actor.sample(state_batch)
        Q1, Q2 = self.critic(state_batch, actor_action)
        actor_loss = -torch.min(Q1, Q2).mean()

        # Optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def save(self, filepath):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
