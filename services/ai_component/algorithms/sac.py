import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional
from algorithms.networks import GaussianPolicy, QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class SAC(BaseAlgorithm):
    def setup(self):
        # Environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_limit = float(self.env.action_space.high[0])  # Assuming symmetric action space

        # Hyperparameters with defaults
        self.gamma: float = 0.99
        self.tau: float = 0.005
        self.alpha: float = 0.2
        self.hidden_size: int = 256
        self.policy_lr: float = 3e-4
        self.q_lr: float = 3e-4
        self.buffer_size: int = 1_000_000
        self.batch_size: int = 256
        self.target_update_interval: int = 1
        self.automatic_entropy_tuning: bool = True
        self.policy_type: str = 'Gaussian'

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

        # Initialize Q-networks and target networks
        self.q1 = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q2 = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q2_target = QNetwork(self.state_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Initialize Policy Network
        self.policy = GaussianPolicy(self.state_dim, self.action_dim, self.hidden_size, self.action_limit).to(self.device)

        # Initialize alpha for entropy tuning
        if self.automatic_entropy_tuning and self.policy_type == "Gaussian":
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.policy_lr)
            self.target_entropy = -self.action_dim  # Common heuristic
            self.alpha = self.log_alpha.exp().item()
        else:
            self.log_alpha = None
            self.alpha_optimizer = None
            self.target_entropy = None

        # Initialize Optimizers for Q-networks and Policy
        self.q1_optimizer = Adam(self.q1.parameters(), lr=self.q_lr)
        self.q2_optimizer = Adam(self.q2.parameters(), lr=self.q_lr)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=self.policy_lr)

        # Initialize update counter
        self.updates = 0

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate:
            # Use deterministic action (mean) during evaluation
            action, _, _ = self.policy.sample_deterministic(state)
        else:
            # Sample action from the policy during training
            action, _, _ = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]

    def train_step(self) -> Optional[Tuple[float, float, float, float, float]]:
        if self.replay_buffer.size < self.batch_size:
            return None  # Not enough samples to train

        # Sample a batch from the replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        # Convert to tensors
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        # Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.policy.sample(next_state_batch)
            q1_next_target = self.q1_target(next_state_batch, next_actions)
            q2_next_target = self.q2_target(next_state_batch, next_actions)
            min_q_next_target = torch.min(q1_next_target, q2_next_target) - self.alpha * next_log_pi
            target_q = reward_batch + (1 - done_batch) * self.gamma * min_q_next_target

        # Current Q estimates
        current_q1 = self.q1(state_batch, action_batch)
        current_q2 = self.q2(state_batch, action_batch)

        # Compute Q-function losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        q_loss = q1_loss + q2_loss

        # Optimize Q1
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        # Optimize Q2
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        # Policy loss
        pi, log_pi, _ = self.policy.sample(state_batch)
        q1_pi = self.q1(state_batch, pi)
        q2_pi = self.q2(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = ((self.alpha * log_pi) - min_q_pi).mean()

        # Optimize Policy
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Entropy temperature adjustment
        alpha_loss_value = 0.0
        if self.automatic_entropy_tuning and self.policy_type == "Gaussian":
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()

        # Soft update of target networks
        if self.updates % self.target_update_interval == 0:
            self.soft_update(self.q1_target, self.q1, self.tau)
            self.soft_update(self.q2_target, self.q2, self.tau)

        self.updates += 1

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss_value, self.alpha

    def soft_update(self, target: torch.nn.Module, source: torch.nn.Module, tau: float):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save(self, filepath: str):
        save_dict = {
            'policy_state_dict': self.policy.state_dict(),
            'q1_state_dict': self.q1.state_dict(),
            'q2_state_dict': self.q2.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
            'q1_optimizer_state_dict': self.q1_optimizer.state_dict(),
            'q2_optimizer_state_dict': self.q2_optimizer.state_dict(),
            'alpha': self.alpha,
            'updates': self.updates,
        }
        if self.automatic_entropy_tuning and self.policy_type == "Gaussian":
            save_dict['log_alpha'] = self.log_alpha.detach().cpu().numpy()
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            save_dict['target_entropy'] = self.target_entropy
        torch.save(save_dict, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.q1.load_state_dict(checkpoint['q1_state_dict'])
        self.q2.load_state_dict(checkpoint['q2_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
        self.q1_optimizer.load_state_dict(checkpoint['q1_optimizer_state_dict'])
        self.q2_optimizer.load_state_dict(checkpoint['q2_optimizer_state_dict'])
        self.alpha = checkpoint.get('alpha', self.alpha)
        self.updates = checkpoint.get('updates', self.updates)

        if self.automatic_entropy_tuning and self.policy_type == "Gaussian":
            # Load log_alpha
            log_alpha = checkpoint.get('log_alpha', None)
            if log_alpha is not None:
                self.log_alpha.data = torch.tensor(log_alpha).to(self.device)
            # Load alpha_optimizer state
            alpha_optimizer_state = checkpoint.get('alpha_optimizer_state_dict', None)
            if alpha_optimizer_state is not None:
                self.alpha_optimizer.load_state_dict(alpha_optimizer_state)
            # Load target_entropy
            self.target_entropy = checkpoint.get('target_entropy', self.target_entropy)
