import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

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
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int, action_limit: float = 1.0):
        super(GaussianPolicy, self).__init__()
        self.action_dim = action_dim
        self.action_limit = action_limit

        # Define the network layers
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Output layers for mean and log_std
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

        # Initialize weights
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor):
        """
        Forward pass to compute mean and log_std of the Gaussian distribution.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # To ensure numerical stability

        return mean, log_std

    def sample(self, state: torch.Tensor):
        """
        Sample an action using the reparameterization trick.
        Returns action, log probability, and the sampled pre-tanh value.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z) * self.action_limit
        log_prob = normal.log_prob(z) - torch.log(self.action_limit * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, z

    def sample_deterministic(self, state: torch.Tensor):
        """
        Select the deterministic action (mean of the Gaussian).
        """
        mean, _ = self.forward(state)
        action = torch.tanh(mean) * self.action_limit
        log_prob = torch.zeros(action.size(0), 1).to(state.device)  # Log prob is zero for deterministic
        return action, log_prob, mean

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int):
        super(QNetwork, self).__init__()
        # Define the network layers
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.q_layer = nn.Linear(hidden_size, 1)

        # Initialize weights
        self.apply(weights_init_)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-value for a state-action pair.
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_value = self.q_layer(x)
        return q_value
    
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
    def __init__(self, state_dim, action_dim, action_limit, hidden_sizes=(256, 256), activation=nn.ReLU):
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
        self.action_limit = action_limit

    def forward(self, state):
        shared_features = self.shared_layers(state)
        action = self.actor_head(shared_features) * self.action_limit
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
