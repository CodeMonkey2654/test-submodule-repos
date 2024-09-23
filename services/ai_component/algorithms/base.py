import abc
import torch

class BaseAlgorithm(abc.ABC):
    def __init__(self, config, env):
        """
        Initialize the base algorithm.

        Parameters:
        - config (dict): Configuration parameters.
        - env (gym.Env): Environment instance.
        """
        self.config = config
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup()

    @abc.abstractmethod
    def setup(self):
        """Set up the algorithm (networks, optimizers, etc.)."""
        pass

    @abc.abstractmethod
    def select_action(self, state):
        """Select an action based on the current state."""
        pass

    @abc.abstractmethod
    def train_step(self):
        """Perform a single training step."""
        pass

    @abc.abstractmethod
    def save(self, filepath):
        """Save the model parameters."""
        pass

    @abc.abstractmethod
    def load(self, filepath):
        """Load the model parameters."""
        pass
