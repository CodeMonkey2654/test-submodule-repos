import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from algorithms.networks import ActorCritic
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym

class A3C(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            action_limit = None
            discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            action_limit = self.env.action_space.high[0]
            discrete = False

        self.actor_critic = ActorCritic(state_dim, action_dim, action_limit).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.config['policy_lr'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.95)  # GAE parameter
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        policy_logits, _ = self.actor_critic(state)
        if evaluate:
            action = torch.argmax(policy_logits).item()
        else:
            dist = F.softmax(policy_logits, dim=-1)
            dist = Categorical(dist)
            action = dist.sample().item()
        return action

    def train_step(self):
        if self.rollout_buffer.size < self.batch_size:
            return  # Not enough samples

        states, actions, rewards, next_states, dones, advantages, returns = self.rollout_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        policy_logits, values = self.actor_critic(states)
        dist = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(dist)

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Actor loss
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss
        critic_loss = F.mse_loss(values.squeeze(), returns)

        # Total loss
        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, filepath):
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
