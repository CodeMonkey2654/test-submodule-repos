import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import GaussianPolicy, QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym


class SAC(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            action_limit = None
        else:
            action_dim = self.env.action_space.shape[0]
            action_limit = self.env.action_space.high[0]

        self.policy = GaussianPolicy(state_dim, action_dim).to(self.device)
        self.q1 = QNetwork(state_dim, action_dim).to(self.device)
        self.q2 = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q2_target = QNetwork(state_dim, action_dim).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config['policy_lr'])
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=self.config['q_lr'])
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=self.config['q_lr'])

        self.replay_buffer = ReplayBuffer(self.config['buffer_size'], state_dim, action_dim)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.005)
        self.alpha = self.config.get('alpha', 0.2)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if evaluate:
            with torch.no_grad():
                mean, std = self.policy(state)
                action = torch.tanh(mean) * self.policy.action_limit
            return action.detach().cpu().numpy()
        else:
            action, _ = self.policy.sample(state)
            return action.detach().cpu().numpy()

    def train_step(self):
        if self.replay_buffer.size < self.config['batch_size']:
            return  # Not enough samples

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.config['batch_size'])

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * q_next

        # Q1 loss
        current_q1 = self.q1(states, actions)
        loss_q1 = torch.nn.functional.mse_loss(current_q1, target_q)

        # Q2 loss
        current_q2 = self.q2(states, actions)
        loss_q2 = torch.nn.functional.mse_loss(current_q2, target_q)

        # Update Q networks
        self.q1_optimizer.zero_grad()
        loss_q1.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        loss_q2.backward()
        self.q2_optimizer.step()

        # Policy loss
        actions_new, log_probs_new = self.policy.sample(states)
        q1_new = self.q1(states, actions_new)
        q2_new = self.q2(states, actions_new)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs_new - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.q1_target.parameters(), self.q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.q2_target.parameters(), self.q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
