import torch
import torch.optim as optim
import torch.nn.functional as F
from algorithms.networks import I2AImaginationNetwork, I2APolicyNetwork
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym


class I2A(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]

        # Initialize networks
        self.imagination = I2AImaginationNetwork(state_dim, action_dim).to(self.device)
        self.policy = I2APolicyNetwork(state_dim, action_dim).to(self.device)

        # Optimizers
        self.imagination_optimizer = optim.Adam(self.imagination.parameters(), lr=self.config['imagination_lr'])
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.config['policy_lr'])

        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim)

        # Hyperparameters
        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)  # GAE parameter
        self.batch_size = self.config.get('batch_size', 256)
        self.imagination_steps = self.config.get('imagination_steps', 5)  # Number of imagination steps

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.policy(state).cpu().numpy()
        return action

    def train_step(self):
        if self.rollout_buffer.size < self.batch_size:
            return  # Not enough samples

        # Sample a batch of real transitions
        states, actions, rewards, next_states, dones = self.rollout_buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # ============================
        # Train Imagination Network
        # ============================

        # Predict next states using the imagination network
        predicted_next_states = self.imagination(states, actions)
        imagination_loss = F.mse_loss(predicted_next_states, next_states)

        # Optimize imagination network
        self.imagination_optimizer.zero_grad()
        imagination_loss.backward()
        self.imagination_optimizer.step()

        # ============================
        # Generate Imagined Trajectories
        # ============================

        imagined_rewards = []
        imagined_dones = []
        imagined_states = []
        imagined_actions = []

        # Initialize imagined states with real next_states
        current_states = predicted_next_states.clone()

        for step in range(self.imagination_steps):
            # Select actions based on current imagined states
            with torch.no_grad():
                imagined_actions_step = self.policy(current_states)
                # Optionally add exploration noise
                # imagined_actions_step += torch.randn_like(imagined_actions_step) * 0.1
                imagined_actions_step = torch.clamp(imagined_actions_step, -1.0, 1.0)
            
            # Predict next states using the imagination network
            imagined_next_states_step = self.imagination(current_states, imagined_actions_step)

            # For simplicity, assume rewards are zero for imagined steps
            # Alternatively, you can train a separate reward model
            imagined_rewards_step = torch.zeros((current_states.size(0), 1)).to(self.device)
            imagined_dones_step = torch.zeros((current_states.size(0), 1)).to(self.device)

            imagined_rewards.append(imagined_rewards_step)
            imagined_dones.append(imagined_dones_step)
            imagined_states.append(current_states)
            imagined_actions.append(imagined_actions_step)

            # Update current states for next imagination step
            current_states = imagined_next_states_step.clone()

        # ============================
        # Train Policy Network
        # ============================

        # Compute returns from imagined rewards
        returns = []
        discounted_sum = torch.zeros((self.batch_size, 1)).to(self.device)
        for step in reversed(range(self.imagination_steps)):
            discounted_sum = imagined_rewards[step] + self.gamma * discounted_sum * (1 - imagined_dones[step])
            returns.insert(0, discounted_sum)

        # Stack imagined states and actions
        all_imagined_states = torch.cat(imagined_states, dim=0)
        all_imagined_actions = torch.cat(imagined_actions, dim=0)
        all_returns = torch.cat(returns, dim=0)

        # Compute policy loss (maximize returns)
        policy_actions = self.policy(all_imagined_states)
        policy_loss = -torch.mean(F.mse_loss(policy_actions, all_imagined_actions, reduction='none') * all_returns)

        # Optimize policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

    def save(self, filepath):
        torch.save({
            'imagination_state_dict': self.imagination.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'imagination_optimizer_state_dict': self.imagination_optimizer.state_dict(),
            'policy_optimizer_state_dict': self.policy_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.imagination.load_state_dict(checkpoint['imagination_state_dict'])
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.imagination_optimizer.load_state_dict(checkpoint['imagination_optimizer_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer_state_dict'])
