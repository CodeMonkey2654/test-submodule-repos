import torch
import torch.optim as optim
import numpy as np
from algorithms.networks import GaussianPolicy, ValueNetwork, CategoricalPolicy
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym


class PPO(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
            self.discrete = True
        else:
            action_dim = self.env.action_space.shape[0]
            self.discrete = False
            self.action_limit = float(self.env.action_space.high[0])

        hidden_sizes = self.config.get('hidden_sizes', (256, 256))
        
        if self.discrete:
            self.policy = CategoricalPolicy(state_dim, action_dim, hidden_sizes=hidden_sizes).to(self.device)
        else:
            self.policy = GaussianPolicy(state_dim, action_dim, hidden_sizes=hidden_sizes, action_limit=self.action_limit).to(self.device)
        
        self.value_net = ValueNetwork(state_dim, hidden_sizes=hidden_sizes).to(self.device)

        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.value_net.parameters()),
                                    lr=self.config['learning_rate'])

        self.buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=self.discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)
        self.eps_clip = self.config.get('eps_clip', 0.2)
        self.K_epochs = self.config.get('K_epochs', 4)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.discrete:
                dist = self.policy(state)
                action = dist.sample() if not evaluate else dist.probs.argmax(dim=-1)
                action_logprob = dist.log_prob(action)
            else:
                action, action_logprob, _ = self.policy.sample(state)
                if evaluate:
                    action = self.policy(state)[0]
        return action.cpu().numpy().squeeze(), action_logprob.cpu().numpy().squeeze()

    def train_step(self):
        states, actions, rewards, next_states, dones, old_logprobs = self.buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device) if not self.discrete else torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_logprobs = torch.FloatTensor(old_logprobs).to(self.device)

        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, dones, states, next_states)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            if self.discrete:
                dist = self.policy(states)
                logprobs = dist.log_prob(actions)
                entropy = dist.entropy()
            else:
                _, logprobs, entropy = self.policy.sample(states, actions)
            
            state_values = self.value_net(states).squeeze()

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

    def compute_gae(self, rewards, dones, states, next_states):
        with torch.no_grad():
            state_values = self.value_net(states).squeeze()
            next_values = self.value_net(next_states).squeeze()
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
            'policy_state_dict': self.policy.state_dict(),
            'value_net_state_dict': self.value_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_net.load_state_dict(checkpoint['value_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
