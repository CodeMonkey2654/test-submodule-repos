import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_(m):
    """
    Initialize network weights using Xavier uniform initialization.
    
    Args:
        m (nn.Module): Module to initialize
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# ============================
# Base Classes
# ============================

class BaseNetwork(nn.Module):
    """
    Base class for all neural networks.
    """
    def __init__(self):
        super(BaseNetwork, self).__init__()
    
    def save(self, path):
        """
        Save the model parameters.
        
        Args:
            path (str): Path to save the model
        """
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """
        Load the model parameters.
        
        Args:
            path (str): Path to load the model from
        """
        self.load_state_dict(torch.load(path))

class BasePolicyNetwork(BaseNetwork):
    """
    Base class for policy networks.
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(BasePolicyNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for _ in range(3):  # Repeat hidden size 3 times
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        output_dim = action_dim
        self.model = nn.Sequential(*layers)
        self.apply(weights_init_)

class BaseValueNetwork(BaseNetwork):
    """
    Base class for value networks.
    """
    def __init__(self, state_dim, hidden_size=256):
        super(BaseValueNetwork, self).__init__()
        layers = []
        input_dim = state_dim
        for _ in range(3):  # Repeat hidden size 3 times
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(nn.ReLU())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))  # Outputs a single scalar value
        self.model = nn.Sequential(*layers)
        self.apply(weights_init_)

# ============================
# Policy Networks
# ============================

class CategoricalPolicy(BasePolicyNetwork):
    """
    Categorical policy network for discrete action spaces.
    Used in: A2C, A3C, PPO, REINFORCE
    """
    def __init__(self, state_dim, action_dim, hidden_size=256):
        super(CategoricalPolicy, self).__init__(state_dim, action_dim, hidden_size)
        self.action_head = nn.Linear(hidden_size, action_dim)
    
    def forward(self, state):
        """
        Forward pass to compute action probabilities.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            torch.distributions.Categorical: Categorical distribution over actions
        """
        x = self.model(state)
        action_probs = F.softmax(self.action_head(x), dim=-1)
        return torch.distributions.Categorical(action_probs)

class GaussianPolicy(BasePolicyNetwork):
    """
    Gaussian policy network for continuous action spaces.
    Used in: DDPG, SAC, PPO (continuous)
    """
    def __init__(self, state_dim, action_dim, hidden_size=256, action_limit=1.0):
        super(GaussianPolicy, self).__init__(state_dim, action_dim, hidden_size)
        self.mean_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Linear(hidden_size, action_dim)
        self.action_limit = action_limit
    
    def forward(self, state):
        """
        Forward pass to compute mean and log_std of the Gaussian distribution.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            tuple: (mean, log_std) of the Gaussian distribution
        """
        x = self.model(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, min=-20, max=2)
        return mean, log_std
    
    def sample(self, state):
        """
        Sample an action using the reparameterization trick.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            tuple: (action, log_prob, pre_tanh_value)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        return self.action_limit * action, log_prob.sum(1, keepdim=True), x_t

# ============================
# Value Networks
# ============================

class ValueNetwork(BaseValueNetwork):
    """
    Value network for estimating state values.
    Used in: A2C, A3C, TRPO
    """
    def forward(self, state):
        """
        Forward pass to compute state value.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            torch.Tensor: Estimated state value
        """
        return self.model(state)

class QNetwork(BaseNetwork):
    """
    Q-Network for estimating state-action values.
    Used in: DQN, DDPG, SAC
    """
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(QNetwork, self).__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """
        Forward pass to compute Q-values.
        
        Args:
            state (torch.Tensor): Input state tensor
            action (torch.Tensor): Input action tensor
        
        Returns:
            tuple: (Q1, Q2) estimated Q-values
        """
        x = torch.cat([state, action], dim=-1)
        
        # Q1 network
        q1 = self.q1[:2](x)  # First two layers
        q1 = self.layer_norm(q1)
        q1 = self.dropout(q1)
        q1 = self.q1[2:](q1)  # Remaining layers
        
        # Q2 network
        q2 = self.q2[:2](x)  # First two layers
        q2 = self.layer_norm(q2)
        q2 = self.dropout(q2)
        q2 = self.q2[2:](q2)  # Remaining layers
        
        return q1, q2

# ============================
# Specialized Networks
# ============================

class I2AImaginationNetwork(BaseNetwork):
    """
    Imagination network for I2A algorithm.
    """
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(I2AImaginationNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, state_dim)
        )
        self.apply(weights_init_)
    
    def forward(self, state, action):
        """
        Forward pass to predict next state.
        
        Args:
            state (torch.Tensor): Current state tensor
            action (torch.Tensor): Action tensor
        
        Returns:
            torch.Tensor: Predicted next state
        """
        x = torch.cat([state, action], dim=-1)
        return self.model(x)

    def predict_n_steps(self, initial_state, actions, n_steps):
        """
        Predict state transitions for n steps.
        
        Args:
            initial_state (torch.Tensor): Initial state tensor
            actions (torch.Tensor): Sequence of action tensors
            n_steps (int): Number of steps to predict
        
        Returns:
            torch.Tensor: Predicted states for n steps
        """
        states = [initial_state]
        for i in range(n_steps):
            next_state = self.forward(states[-1], actions[i])
            states.append(next_state)
        return torch.stack(states[1:])  # Exclude initial state

class I2APolicyNetwork(BasePolicyNetwork):
    """
    Policy network for I2A algorithm.
    """
    def __init__(self, state_dim, action_dim, hidden_size=512):
        super(I2APolicyNetwork, self).__init__(state_dim, action_dim, hidden_size)
        self.apply(weights_init_)

    def forward(self, state):
        """
        Forward pass to compute action probabilities.
        
        Args:
            state (torch.Tensor): Input state tensor
        
        Returns:
            torch.distributions.Categorical: Categorical distribution over actions
        """
        return self.model(state)
