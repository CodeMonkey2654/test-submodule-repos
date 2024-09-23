from .a2c import A2C
from .a3c import A3C
from .cem import CEM
from .ddpg import DDPG
from .dqn import DQN
from .evolutionary_strategies import EvolutionStrategies
from .i2a import I2A
from .ppo import PPO
from .q_learning import QLearning
from .reinforce import REINFORCE
from .sac import SAC
from .sarsa import SARSA
from .td3 import TD3
from .trpo import TRPO
from .soft_q_learning import SoftQLearning
from .base import BaseAlgorithm


__all__ = [
    'A2C',
    'A3C',
    'CEM',
    'DDPG',
    'DQN',
    'EvolutionStrategies',
    'I2A',
    'PPO',
    'QLearning',
    'REINFORCE',
    'SAC',
    'SARSA',
    'TD3',
    'TRPO',
    'SoftQLearning',
    'BaseAlgorithm'
]

