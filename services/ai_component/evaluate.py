import argparse
import yaml
import os
import gymnasium as gym
import torch
from algorithms.ppo.ppo import PPO
from algorithms.sac.sac import SAC
from algorithms.ddpg.ddpg import DDPG
from utils.logger import Logger
from utils.helpers import set_seed
from environments import register_environments

def get_algorithm(name):
    if name.lower() == 'ppo':
        return PPO
    elif name.lower() == 'sac':
        return SAC
    elif name.lower() == 'ddpg':
        return DDPG
    else:
        raise NotImplementedError(f"Algorithm {name} is not implemented.")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained reinforcement learning agent.")
    parser.add_argument('--env', type=str, default='LunarRegolith-v0', help='Gym environment name')
    parser.add_argument('--algorithm', type=str, required=True, choices=['ppo', 'sac', 'ddpg'], help='RL algorithm used during training')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--log_dir', type=str, default='evaluation_logs', help='Directory to save evaluation logs')
    parser.add_argument('--episodes', type=int, default=100, help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for evaluation')
    parser.add_argument('--robot_sdl_path', type=str, required=True, help='Path to the robot SDL file')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')

    args = parser.parse_args()

    # Load configuration (if needed)
    # For evaluation, configuration parameters should match those used during training
    # Assuming the same config file was used, but you can also pass it as an argument if needed
    # Here, we'll assume it's part of the model's saved state or similar

    # Register environments
    register_environments()

    # Create environment with necessary parameters
    # Assuming that during training, the same parameters were used. If not, you may need to pass them
    # Alternatively, you can load them from a config file or model checkpoint
    # Here, we'll require the user to provide a config file similar to training
    # For simplicity, we'll add optional arguments to specify them
    # Alternatively, extend the model saving to include env parameters

    # For this example, we'll assume default parameters or require a config file
    # Here, we'll prompt the user to specify a config file or set defaults
    # To keep it simple, we'll set defaults similar to the training script

    # Load a default or specific config
    # Here, we'll assume the config is not strictly necessary for evaluation,
    # but to ensure environment parameters are consistent, you might need them.

    # If you have saved env parameters during training, load them here
    # For this example, we'll proceed with default values

    # Create environment
    # Note: In practice, it's essential to ensure that the environment's parameters
    # match those used during training for accurate evaluation

    # To handle this properly, you might consider saving the config along with the model
    # and loading it here. For now, let's proceed with defaults, but be aware of this caveat.

    # If you have saved a config during training, you can load it as follows:
    # with open(config_path, 'r') as f:
    #     config = yaml.safe_load(f)
    # And pass the parameters to `gym.make`

    # For simplicity, we will require the user to provide a config file during evaluation
    # similar to training.

    # Adding a config argument
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file used during training')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create environment
    env = gym.make(args.env,
                   robot_sdl_path=args.robot_sdl_path,
                   arena_length=config.get('arena_length', 20.0),
                   arena_width=config.get('arena_width', 20.0),
                   material_friction=config.get('material_friction', 1.5),
                   material_density=config.get('material_density', 2.0),
                   num_craters=config.get('num_craters', 8),
                   num_boulders=config.get('num_boulders', 15),
                   action_type=config.get('action_type', 'continuous'),
                   goal_area_size=config.get('goal_area_size', (2.0, 2.0, 1.0))
                  )

    # Set seed for reproducibility
    set_seed(env, args.seed)

    # Initialize algorithm
    AlgorithmClass = get_algorithm(args.algorithm)
    agent = AlgorithmClass(config, env)

    # Load the trained model
    agent.load(args.model_path)
    print(f"Loaded model from {args.model_path}")

    # Initialize logger
    logger = Logger(args.log_dir)

    for episode in range(1, args.episodes + 1):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (done or truncated):
            # Select action
            if args.algorithm.lower() in ['sac', 'ddpg']:
                action = agent.select_action(state, evaluate=True)
            else:
                action, _ = agent.select_action(state)

            # Take action
            next_state, reward, done, truncated, info = env.step(action)

            state = next_state
            episode_reward += reward
            steps += 1

            # Render if required
            if args.render:
                env.render()

        # Log episode reward
        logger.log('episode_reward', episode_reward)
        logger.log('steps', steps)

        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(logger.logs['episode_reward'][-10:]) / 10
            avg_steps = sum(logger.logs['steps'][-10:]) / 10
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")

    # Save evaluation logs
    logger.save_logs()
    logger.plot_logs()

    env.close()

if __name__ == "__main__":
    main()