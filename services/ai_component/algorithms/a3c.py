import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.distributions import Categorical
from algorithms.networks import CategoricalPolicy, ValueNetwork
from algorithms.base import BaseAlgorithm
import gymnasium as gym
import numpy as np

class A3CWorker(mp.Process):
    def __init__(self, global_actor, global_critic, optimizer, global_episode, config, worker_id, env):
        super(A3CWorker, self).__init__()
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.optimizer = optimizer
        self.global_episode = global_episode
        self.worker_id = worker_id
        self.env = env
        self.config = config
        self.local_actor = CategoricalPolicy(self.env.observation_space.shape[0], self.env.action_space.n).to(self.config['device'])
        self.local_critic = ValueNetwork(self.env.observation_space.shape[0]).to(self.config['device'])
        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.95)  # GAE parameter
        self.max_episode_length = self.config.get('max_episode_length', 200)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.config['device'])
        dist = self.local_actor(state)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action).to(self.config['device']))
        return action, log_prob

    def compute_gae(self, rewards, values, next_value, dones):
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantage = delta + self.gamma * self.tau * (1 - dones[t]) * advantage
            advantages.insert(0, advantage)
            next_value = values[t]
        return advantages

    def train_step(self, states, actions, rewards, dones, log_probs):
        states = torch.FloatTensor(states).to(self.config['device'])
        actions = torch.LongTensor(actions).to(self.config['device'])
        rewards = torch.FloatTensor(rewards).to(self.config['device'])
        dones = torch.FloatTensor(dones).to(self.config['device'])

        dist = self.local_actor(states)
        values = self.local_critic(states).squeeze()

        next_state = torch.FloatTensor(self.env.reset()[0]).to(self.config['device'])
        with torch.no_grad():
            next_value = self.local_critic(next_state).squeeze()

        returns = rewards + self.gamma * next_value * (1 - dones)

        advantages = self.compute_gae(rewards, values, next_value, dones)
        advantages = torch.FloatTensor(advantages).to(self.config['device'])

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages).mean()

        critic_loss = F.mse_loss(values, returns)

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.local_actor.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(self.local_critic.parameters(), max_norm=0.5)

        for local_param, global_param in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_param._grad = local_param.grad
        for local_param, global_param in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_param._grad = local_param.grad

        self.optimizer.step()

        self.local_actor.load_state_dict(self.global_actor.state_dict())
        self.local_critic.load_state_dict(self.global_critic.state_dict())

    def run(self):
        while self.global_episode.value < self.config['max_episodes']:
            state, _ = self.env.reset()
            states, actions, rewards, dones, log_probs = [], [], [], [], []

            for _ in range(self.max_episode_length):
                action, log_prob = self.select_action(state)
                next_state, reward, done, _, _ = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob)

                state = next_state

                if done:
                    break

            self.train_step(states, actions, rewards, dones, log_probs)

            with self.global_episode.get_lock():
                self.global_episode.value += 1
                print(f"Worker {self.worker_id}, Global Episode {self.global_episode.value}")

class A3C(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.global_actor = CategoricalPolicy(state_dim, action_dim).to(self.device)
        self.global_critic = ValueNetwork(state_dim).to(self.device)
        self.global_actor.share_memory()
        self.global_critic.share_memory()
        self.optimizer = optim.Adam(list(self.global_actor.parameters()) + list(self.global_critic.parameters()), lr=self.config['policy_lr'])
        self.global_episode = mp.Value('i', 0)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(self.device)
        dist = self.global_actor(state)
        action = dist.sample().item()
        return action

    def train_step(self):
        workers = [A3CWorker(self.global_actor, self.global_critic, self.optimizer, self.global_episode, self.config, worker_id, self.env) for worker_id in range(self.config['num_workers'])]
        
        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

    def save(self, filepath):
        torch.save({
            'global_actor_state_dict': self.global_actor.state_dict(),
            'global_critic_state_dict': self.global_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.global_actor.load_state_dict(checkpoint['global_actor_state_dict'])
        self.global_critic.load_state_dict(checkpoint['global_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
