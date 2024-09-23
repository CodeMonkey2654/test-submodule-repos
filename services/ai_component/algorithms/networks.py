import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================
# Base Classes (Optional)
# ============================

class BasePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(BasePolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        self.model = nn.Sequential(*layers)

class BaseValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(256, 256)):
        super(BaseValueNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))  # Outputs a single scalar value
        self.model = nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(512, 512)):
        super(GaussianPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Tanh()
        )

    def forward(self, state):
        return self.model(state)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(512, 512)):
        super(QNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
    
class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_sizes=(256, 256)):
        super(ValueNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, state):
        return self.model(state)
    
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(PolicyNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_dim)
        )

    def forward(self, state):
        return self.model(state)


# ============================
# Actor-Critic Networks
# ============================

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), activation=nn.ReLU, output_activation=nn.Tanh):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(output_activation())
        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(activation())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
    
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
        super(ActorCritic, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_sizes[0]),
            activation(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            activation()
        )
        self.actor_head = nn.Sequential(
            nn.Linear(hidden_sizes[1], action_dim),
            nn.Tanh()
        )
        self.critic_head = nn.Linear(hidden_sizes[1], 1)

    def forward(self, state):
        shared_features = self.shared_layers(state)
        action = self.actor_head(shared_features)
        value = self.critic_head(shared_features)
        return action, value
    

# ============================
# SoftQ Networks
# ============================


class SoftQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(SoftQNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.model(x)
    
    
# ============================
# TRPO Networks
# ============================

class TRPOPolicyNetwork(BasePolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), discrete=False):
        super(TRPOPolicyNetwork, self).__init__(state_dim, action_dim, hidden_sizes)
        self.discrete = discrete
        if self.discrete:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state):
        x = self.model(state)
        if self.discrete:
            action_logits = self.action_head(x)
            return action_logits
        else:
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, min=-20, max=2)  # Stability
            std = torch.exp(log_std)
            return mean, std

class TRPOValueNetwork(BaseValueNetwork):
    def __init__(self, state_dim, hidden_sizes=(256, 256)):
        super(TRPOValueNetwork, self).__init__(state_dim, hidden_sizes)

    def forward(self, state):
        return self.model(state)

# ============================
# CEM Networks
# ============================

class CEMPolicyNetwork(BasePolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_sizes=(64, 64), discrete=False):
        super(CEMPolicyNetwork, self).__init__(state_dim, action_dim, hidden_sizes)
        self.discrete = discrete
        if self.discrete:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state):
        x = self.model(state)
        if self.discrete:
            action_logits = self.action_head(x)
            return action_logits
        else:
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            return mean, std

# ============================
# I2A Networks
# ============================

class I2AImaginationNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256)):
        super(I2AImaginationNetwork, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, state_dim))  # Predict next state
        self.model = nn.Sequential(*layers)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        next_state = self.model(x)
        return next_state

class I2APolicyNetwork(BasePolicyNetwork):
    def __init__(self, state_dim, action_dim, hidden_sizes=(256, 256), discrete=False):
        super(I2APolicyNetwork, self).__init__(state_dim, action_dim, hidden_sizes)
        self.discrete = discrete
        if self.discrete:
            self.action_head = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.mean_head = nn.Linear(hidden_sizes[-1], action_dim)
            self.log_std_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, state):
        x = self.model(state)
        if self.discrete:
            action_logits = self.action_head(x)
            return action_logits
        else:
            mean = self.mean_head(x)
            log_std = self.log_std_head(x)
            log_std = torch.clamp(log_std, min=-20, max=2)
            std = torch.exp(log_std)
            return mean, std
