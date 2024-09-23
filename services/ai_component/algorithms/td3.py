import torch
import torch.nn.functional as F
from torch.optim import Adam
from algorithms.networks import GaussianPolicy, QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import numpy as np

class TD3(BaseAlgorithm):
    def setup(self):
        # Environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.max_action = float(self.env.action_space.high[0])

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.policy_noise = self.config.get('policy_noise', 0.2)
        self.noise_clip = self.config.get('noise_clip', 0.5)
        self.policy_freq = self.config.get('policy_freq', 2)
        self.lr = self.config.get('lr', 3e-4)
        self.buffer_size = self.config.get('buffer_size', 1000000)
        self.batch_size = self.config.get('batch_size', 256)
        self.hidden_size = self.config.get('hidden_size', 256)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor (policy) network
        self.actor = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size, self.max_action).to(self.device)
        self.actor_target = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)

        # Initialize critic (Q-value) networks
        self.critic1 = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic2 = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic1_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic2_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = Adam(self.critic1.parameters(), lr=self.lr)
        self.critic2_optimizer = Adam(self.critic2.parameters(), lr=self.lr)

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

        self.total_it = 0

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        if evaluate:
            action, _, _ = self.actor.sample(state)
        else:
            action, _, _ = self.actor.sample(state)
            action = action + torch.randn_like(action) * self.policy_noise
            action = action.clamp(-self.max_action, self.max_action)
        return action.cpu().data.numpy().flatten()

    def train_step(self):
        self.total_it += 1

        # Sample from replay buffer
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)

        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(1)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).to(self.device).unsqueeze(1)

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action, _, _ = self.actor_target.sample(next_state)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)

            # Compute the target Q value
            target_Q1, _ = self.critic1_target(next_state, next_action)
            target_Q2, _ = self.critic2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1 - done) * self.gamma * target_Q

        # Get current Q estimates
        current_Q1, _ = self.critic1(state, action)
        current_Q2, _ = self.critic2(state, action)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the critic
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss
            actor_loss = -self.critic1(state, self.actor(state)[0])[0].mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filename):
        torch.save(self.critic1.state_dict(), filename + "_critic1")
        torch.save(self.critic1_optimizer.state_dict(), filename + "_critic1_optimizer")
        
        torch.save(self.critic2.state_dict(), filename + "_critic2")
        torch.save(self.critic2_optimizer.state_dict(), filename + "_critic2_optimizer")
        
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic1.load_state_dict(torch.load(filename + "_critic1"))
        self.critic1_optimizer.load_state_dict(torch.load(filename + "_critic1_optimizer"))
        self.critic1_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2.load_state_dict(torch.load(filename + "_critic2"))
        self.critic2_optimizer.load_state_dict(torch.load(filename + "_critic2_optimizer"))
        self.critic2_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size, self.max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
