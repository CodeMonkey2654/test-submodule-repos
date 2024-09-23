import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from algorithms.networks import CategoricalPolicy, GaussianPolicy, ValueNetwork
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym


class A2C(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            self.actor = CategoricalPolicy(state_dim, action_dim).to(self.device)
            self.discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            action_limit = self.env.action_space.high[0]
            self.actor = GaussianPolicy(state_dim, action_dim, action_limit=action_limit).to(self.device)
            self.discrete = False

        self.critic = ValueNetwork(state_dim).to(self.device)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=self.config['policy_lr'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=self.discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.95)  # GAE parameter
        self.batch_size = self.config.get('batch_size', 256)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        if self.discrete:
            dist = self.actor(state)
            if evaluate:
                action = torch.argmax(dist.probs).item()
            else:
                action = dist.sample().item()
            log_prob = dist.log_prob(torch.tensor(action)).item()
        else:
            mean, log_std = self.actor(state)
            if evaluate:
                action = mean.detach()
            else:
                action, log_prob = self.actor.sample(state)
                log_prob = log_prob.item()
            action = action.cpu().numpy()
        return action, log_prob

    def compute_gae(self, rewards, values, next_values, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.tau * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
        return advantages

    def train_step(self):
        if self.rollout_buffer.size < self.batch_size:
            return  # Not enough samples to train
    
        # Sample a batch from the buffer
        states, actions, rewards, next_states, dones, _ = self.rollout_buffer.sample(self.batch_size)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Get current values from the critic for the sampled states
        values = self.critic(states)

        # Get values for the next states from the critic
        with torch.no_grad():
            next_values = self.critic(next_states)

        # Compute bootstrapped returns
        returns = rewards + self.gamma * next_values.squeeze() * (1 - dones)

        # Compute advantages (returns - values)
        advantages = returns - values.squeeze()

        # Actor loss
        if self.discrete:
            dist = self.actor(states)
            log_probs = dist.log_prob(actions.long().squeeze())
            entropy = dist.entropy().mean()
        else:
            mean, log_std = self.actor(states)
            dist = torch.distributions.Normal(mean, log_std.exp())
            log_probs = dist.log_prob(actions).sum(1)
            entropy = dist.entropy().sum(1).mean()

        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic loss (use MSE between the predicted values and bootstrapped returns)
        critic_loss = F.mse_loss(values.squeeze(), returns.detach())

        # Total loss: actor loss + critic loss + entropy regularization
        entropy_coeff = self.config.get('entropy_coeff', 0.001)  # Use a configurable entropy coefficient
        loss = actor_loss + 0.5 * critic_loss - entropy_coeff * entropy

        # Optimize the network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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
