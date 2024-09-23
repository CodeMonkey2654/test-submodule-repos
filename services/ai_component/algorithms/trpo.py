import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from algorithms.networks import TRPOPolicyNetwork, TRPOValueNetwork
from utils.replay_buffer import RolloutBuffer
from .base import BaseAlgorithm
import gymnasium as gym

class TRPO(BaseAlgorithm):
    def setup(self):
        state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
            discrete = False
        else:
            action_dim = self.env.action_space.n
            discrete = True

        self.policy = TRPOPolicyNetwork(state_dim, action_dim).to(self.device)
        self.value_network = TRPOValueNetwork(state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.config['value_lr'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], state_dim, action_dim, discrete=discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)  # GAE parameter
        self.max_kl = self.config.get('max_kl', 1e-3)
        self.cg_damping = self.config.get('cg_damping', 1e-2)
        self.cg_iterations = self.config.get('cg_iterations', 10)
        self.backtracking_coeff = self.config.get('backtracking_coeff', 0.8)
        self.backtracking_steps = self.config.get('backtracking_steps', 10)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            mean, std = self.policy(state)
            if evaluate:
                action = mean.cpu().numpy()
            else:
                dist = Normal(mean, std)
                action = dist.sample().cpu().numpy()
        return action

    def train_step(self):
        if self.rollout_buffer.size < self.config['batch_size']:
            return  # Not enough samples

        # Sample a batch of trajectories
        states, actions, rewards, next_states, dones = self.rollout_buffer.sample(self.config['batch_size'])
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute advantage estimates using GAE
        with torch.no_grad():
            values = self.value_network(states).squeeze(1)
            next_values = self.value_network(next_states).squeeze(1)
            deltas = rewards.squeeze(1) + self.gamma * next_values * (1 - dones.squeeze(1)) - values
            advantages = torch.zeros_like(rewards.squeeze(1)).to(self.device)
            gae = 0
            for step in reversed(range(len(rewards))):
                gae = deltas[step] + self.gamma * self.lam * (1 - dones[step]) * gae
                advantages[step] = gae
            returns = advantages + values

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Compute the surrogate loss
        old_mean, old_std = self.policy(states)
        old_dist = Normal(old_mean.detach(), old_std.detach())
        old_log_probs = old_dist.log_prob(actions).sum(dim=1).detach()

        # Define the surrogate loss function
        def surrogate_loss():
            mean, std = self.policy(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            ratio = torch.exp(log_probs - old_log_probs)
            return (ratio * advantages).mean()

        # Compute the policy gradient
        surrogate = surrogate_loss()
        surrogate.backward()
        policy_grad = self.get_flat_grad(surrogate_loss, self.policy).detach()

        # Compute the Fisher vector product
        def fisher_vector_product(vector):
            mean, std = self.policy(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            kl = self.compute_kl_divergence(mean, std, old_mean.detach(), old_std.detach())
            kl = kl.mean()
            kl.backward(create_graph=True)
            grads = self.get_flat_grad(lambda: kl, self.policy)
            fisher = self.flatten_params([param.grad for param in self.policy.parameters()])
            return fisher + vector * self.cg_damping

        # Solve for step direction using conjugate gradient
        step_direction = self.conjugate_gradient(policy_grad, fisher_vector_product)

        # Compute step size
        step_size = torch.dot(policy_grad, step_direction) / (torch.dot(step_direction, fisher_vector_product(step_direction)) + 1e-8)
        step = step_direction * (2 * self.max_kl / (torch.dot(step_direction, fisher_vector_product(step_direction)) + 1e-8))

        # Perform line search
        success = False
        flat_params = self.get_flat_params()
        for _ in range(self.backtracking_steps):
            new_params = flat_params + self.backtracking_coeff * step
            self.set_flat_params(new_params)
            new_surrogate = surrogate_loss()
            # Compute new KL divergence
            mean_new, std_new = self.policy(states)
            kl = self.compute_kl_divergence(mean_new, std_new, old_mean.detach(), old_std.detach()).mean().item()
            if new_surrogate > surrogate.item() and kl <= self.max_kl:
                success = True
                break
            self.set_flat_params(flat_params)
            self.backtracking_coeff *= 0.5
        if not success:
            print("Line search failed.")

        # Update value function
        value_loss = F.mse_loss(self.value_network(states).squeeze(1), returns)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

    def conjugate_gradient(self, b, A, max_iterations=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = torch.dot(r, r)
        for _ in range(max_iterations):
            Ap = A(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-8)
            x += alpha * p
            r -= alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < residual_tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        return x

    def compute_kl_divergence(self, mean1, std1, mean2, std2):
        var1 = std1.pow(2)
        var2 = std2.pow(2)
        kl = torch.log(std2 / std1) + (var1 + (mean1 - mean2).pow(2)) / (2 * var2) - 0.5
        return kl.sum(dim=1)

    def get_flat_grad(self, loss_func, model):
        grads = torch.autograd.grad(loss_func(), model.parameters(), retain_graph=True)
        return torch.cat([grad.view(-1) for grad in grads])

    def flatten_params(self, params):
        return torch.cat([param.view(-1) for param in params])

    def get_flat_params(self):
        return torch.cat([param.view(-1) for param in self.policy.parameters()])

    def set_flat_params(self, flat_params):
        pointer = 0
        for param in self.policy.parameters():
            num_params = param.numel()
            param.data.copy_(flat_params[pointer:pointer + num_params].view(param.size()))
            pointer += num_params

    def save(self, filepath):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'value_network_state_dict': self.value_network.state_dict(),
            'value_optimizer_state_dict': self.value_optimizer.state_dict(),
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.value_network.load_state_dict(checkpoint['value_network_state_dict'])
        self.value_optimizer.load_state_dict(checkpoint['value_optimizer_state_dict'])
