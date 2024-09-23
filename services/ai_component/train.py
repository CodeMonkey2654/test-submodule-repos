import argparse
import yaml
import os
import gymnasium as gym
import torch
from algorithms.ppo import PPO
from algorithms.sac import SAC
from algorithms.ddpg import DDPG
from utils.logger import Logger


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
    parser = argparse.ArgumentParser(description="Train a reinforcement learning agent.")
    parser.add_argument('--env', type=str, default='Humanoid-v4', help='Gym environment name')
    parser.add_argument('--algorithm', type=str, required=True, choices=['ppo', 'sac', 'ddpg'], help='RL algorithm to use')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML config file')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save logs and models')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of training episodes')
    parser.add_argument('--model_path', type=str, default=None, help='Path to a pre-trained model to load')
    parser.add_argument('--robot_sdl_path', type=str, required=False, help='Path to the robot SDL file')

    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)


    # Initialize environment
    env = gym.make(args.env)

    # Initialize algorithm
    AlgorithmClass = get_algorithm(args.algorithm)
    agent = AlgorithmClass(config, env)

    # Load pre-trained model if specified
    if args.model_path is not None:
        agent.load(args.model_path)
        print(f"Loaded pre-trained model from {args.model_path}")

    # Initialize logger
    logger = Logger(args.log_dir)

    for episode in range(args.episodes):
        state, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0.0
        steps = 0

        while not (done or truncated):
            if isinstance(agent, (SAC, DDPG)):
                action = agent.select_action(state)
            else:
                action, logprob = agent.select_action(state)

            next_state, reward, done, truncated, info = env.step(action)

            if isinstance(agent, PPO):
                agent.buffer.add(state, action, reward, truncated or done, logprob)
            else:
                agent.replay_buffer.add(state, action, reward, next_state, truncated or done)

            state = next_state
            episode_reward += reward
            steps += 1

        # Log episode results
        logger.log('episode_reward', episode_reward)
        logger.log('steps', steps)

        # Perform training step
        if isinstance(agent, PPO):
            agent.train_step()
        else:
            for _ in range(config.get('train_iterations', 1)):
                agent.train_step()

        # Print progress and save logs/model periodically
        if (episode + 1) % 10 == 0:
            avg_reward = sum(logger.logs['episode_reward'][-10:]) / 10
            avg_steps = sum(logger.logs['steps'][-10:]) / 10
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}, Average Steps: {avg_steps:.2f}")
            logger.save_logs()

        if (episode + 1) % 100 == 0:
            save_path = os.path.join(args.log_dir, f"{args.algorithm}_episode_{episode + 1}.pth")
            agent.save(save_path)
            print(f"Model saved at {save_path}")

    # Save final model
    final_model_path = os.path.join(args.log_dir, f"{args.algorithm}_final.pth")
    agent.save(final_model_path)
    print(f"Final model saved at {final_model_path}")

    env.close()
    logger.plot_logs()

if __name__ == "__main__":
    main()
