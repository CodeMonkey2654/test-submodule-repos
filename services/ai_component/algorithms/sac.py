import torch
import torch.nn.functional as F
from torch.optim import Adam
from typing import Tuple, Optional
from algorithms.networks import GaussianPolicy, QNetwork
from utils.replay_buffer import ReplayBuffer
from algorithms.base import BaseAlgorithm

class SAC(BaseAlgorithm):
    def setup(self):
        # Environment dimensions
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_limit = float(self.env.action_space.high[0])

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.001)
        self.alpha = self.config.get('alpha', 0.2)
        self.hidden_size = self.config.get('hidden_size', 512)
        self.lr = self.config.get('lr', 1e-3)
        self.buffer_size = self.config.get('buffer_size', 1000000)
        self.batch_size = self.config.get('batch_size', 512)
        self.target_update_interval = self.config.get('target_update_interval', 2)
        self.automatic_entropy_tuning = self.config.get('automatic_entropy_tuning', True)
        self.n_step = self.config.get('n_step', 3)  
        self.prioritized_replay = self.config.get('prioritized_replay', True)
        self.grad_clip = self.config.get('grad_clip', 1.0)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Replay Buffer
        self.replay_buffer = ReplayBuffer(self.buffer_size, self.state_dim, self.action_dim)

        # Initialize Q-networks and target networks
        self.critic = QNetwork(self.state_dim, self.action_dim, hidden_size=self.hidden_size).to(self.device)
        self.critic_target = QNetwork(self.state_dim, self.action_dim, hidden_size=self.hidden_size).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Initialize Policy Network
        self.actor = GaussianPolicy(self.state_dim, self.action_dim, hidden_size=self.hidden_size, action_limit=self.action_limit).to(self.device)

        # Initialize optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.lr)

        # Automatic entropy tuning
        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = self.config.get('alpha', 0.2)

        self.total_steps = 0

    def select_action(self, state: torch.Tensor, evaluate: bool = False) -> torch.Tensor:
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        with torch.no_grad():
            if evaluate:
                action, _, _ = self.actor.sample(state)
            else:
                action, _, _ = self.actor.sample(state)
        return action.detach().cpu().numpy()[0]

    def train_step(self) -> Optional[Tuple[float, float, float, float]]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        done_batch = torch.FloatTensor(done_batch).to(self.device).unsqueeze(1)
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target

        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1.squeeze(), next_q_value.squeeze())
        qf2_loss = F.mse_loss(qf2.squeeze(), next_q_value.squeeze())
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            torch.nn.utils.clip_grad_norm_([self.log_alpha], self.grad_clip)
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)

        if self.total_steps % self.target_update_interval == 0:
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        self.total_steps += 1

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item()

    def save(self, filepath: str):
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.automatic_entropy_tuning else None,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.automatic_entropy_tuning else None,
            'total_steps': self.total_steps
        }, filepath)

    def load(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        if self.automatic_entropy_tuning:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        self.total_steps = checkpoint['total_steps']
