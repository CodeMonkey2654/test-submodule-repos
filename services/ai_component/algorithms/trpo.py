import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
from algorithms.networks import GaussianPolicy, ValueNetwork
from utils.replay_buffer import RolloutBuffer
from algorithms.base import BaseAlgorithm
import gymnasium as gym
import numpy as np

class TRPO(BaseAlgorithm):
    def setup(self):
        self.state_dim = self.env.observation_space.shape[0]
        if isinstance(self.env.action_space, gym.spaces.Box):
            self.action_dim = self.env.action_space.shape[0]
            self.discrete = False
        else:
            self.action_dim = self.env.action_space.n
            self.discrete = True

        self.policy = GaussianPolicy(self.state_dim, self.action_dim).to(self.device)
        self.value_network = ValueNetwork(self.state_dim).to(self.device)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=self.config['value_lr'])

        self.rollout_buffer = RolloutBuffer(self.config['buffer_size'], self.state_dim, self.action_dim, discrete=self.discrete)

        self.gamma = self.config.get('gamma', 0.99)
        self.tau = self.config.get('tau', 0.95)  # GAE parameter
        self.max_kl = self.config.get('max_kl', 0.01)
        self.damping = self.config.get('damping', 0.1)
        self.l2_reg = self.config.get('l2_reg', 1e-3)
        self.value_train_iters = self.config.get('value_train_iters', 80)
        self.batch_size = self.config.get('batch_size', 64)

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.discrete:
                probs = self.policy(state)
                if evaluate:
                    action = probs.argmax().item()
                else:
                    action = torch.distributions.Categorical(probs).sample().item()
            else:
                mean, log_std = self.policy(state)
                if evaluate:
                    action = mean.cpu().numpy()[0]
                else:
                    action, _ = self.policy.sample(state)
                    action = action.cpu().numpy()[0]
        return action

    def train_step(self):
        if self.rollout_buffer.size < self.batch_size:
            return

        states, actions, rewards, next_states, dones, log_probs = self.rollout_buffer.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)

        # Compute advantages and returns
        with torch.no_grad():
            values = self.value_network(states)
            next_values = self.value_network(next_states)
            
            advantages = torch.zeros_like(rewards).to(self.device)
            returns = torch.zeros_like(rewards).to(self.device)
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.tau * (1 - dones[t]) * gae
                advantages[t] = gae
                returns[t] = gae + values[t]

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Update value function
        for _ in range(self.value_train_iters):
            value_loss = F.mse_loss(self.value_network(states), returns)
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        # Compute policy loss and KL divergence
        def compute_loss_and_kl():
            if self.discrete:
                new_probs = self.policy(states)
                new_log_probs = torch.log(new_probs.gather(1, actions.long()))
            else:
                new_mean, new_log_std = self.policy(states)
                new_std = torch.exp(new_log_std)
                new_dist = Normal(new_mean, new_std)
                new_log_probs = new_dist.log_prob(actions).sum(dim=1, keepdim=True)

            ratio = torch.exp(new_log_probs - old_log_probs)
            surrogate_loss = (ratio * advantages).mean()

            if self.discrete:
                old_probs = torch.exp(old_log_probs)
                kl = (old_probs * (torch.log(old_probs) - torch.log(new_probs))).sum(1).mean()
            else:
                old_mean, old_log_std = self.policy(states)
                old_std = torch.exp(old_log_std)
                kl = torch.distributions.kl_divergence(Normal(old_mean, old_std), Normal(new_mean, new_std)).sum(1).mean()

            return surrogate_loss, kl

        # Compute gradient of surrogate loss
        loss, _ = compute_loss_and_kl()
        grads = torch.autograd.grad(loss, self.policy.parameters())
        loss_grad = torch.cat([grad.view(-1) for grad in grads]).detach()

        # Compute Fisher-vector product
        def Fvp(v):
            kl = compute_loss_and_kl()[1]
            kl_grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
            kl_grad = torch.cat([grad.view(-1) for grad in kl_grads])
            Fvp = torch.autograd.grad(kl_grad @ v, self.policy.parameters())
            Fvp = torch.cat([grad.contiguous().view(-1) for grad in Fvp]).detach()
            return Fvp + self.damping * v

        # Compute step direction using conjugate gradient
        step_dir = self.conjugate_gradient(Fvp, loss_grad)

        # Compute step size
        shs = 0.5 * (step_dir @ Fvp(step_dir))
        lm = torch.sqrt(shs / self.max_kl)
        fullstep = step_dir / lm

        # Perform line search
        old_params = torch.cat([param.data.view(-1) for param in self.policy.parameters()])
        success, new_params = self.line_search(compute_loss_and_kl, old_params, fullstep)

        if success:
            index = 0
            for param in self.policy.parameters():
                param_size = param.numel()
                param.data.copy_(new_params[index:index + param_size].view(param.size()))
                index += param_size
        else:
            print("Line search failed. Not updating policy.")

        self.rollout_buffer.clear()

    def conjugate_gradient(self, Avp_func, b, nsteps=10, residual_tol=1e-10):
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rdotr = r @ r
        for i in range(nsteps):
            Avp = Avp_func(p)
            alpha = rdotr / (p @ Avp)
            x += alpha * p
            r -= alpha * Avp
            new_rdotr = r @ r
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    def line_search(self, f, x, fullstep, max_backtracks=10, accept_ratio=0.1):
        fval, kl = f()
        expected_improve = (- fullstep * fval).sum()
        for stepfrac in [1.0, 0.5, 0.25, 0.1, 0.05, 0.01, 0.005, 0.001]:
            xnew = x + stepfrac * fullstep
            self.set_params(xnew)
            newfval, new_kl = f()
            actual_improve = fval - newfval
            expected_improve_rate = expected_improve * stepfrac
            ratio = actual_improve / expected_improve_rate

            if ratio > accept_ratio and new_kl <= self.max_kl:
                return True, xnew

        self.set_params(x)
        return False, x

    def set_params(self, flat_params):
        prev_ind = 0
        for param in self.policy.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size

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
