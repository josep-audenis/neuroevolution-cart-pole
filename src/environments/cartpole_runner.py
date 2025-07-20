import gymnasium as gym
import numpy as np
from evolution.neural_network import NeuralNetwork

def evaluate_genome(genome, input_size, hidden_size, output_size, render=False):
    env = gym.make("CartPole-v1", render_mode="human" if render else None)
    obs, _ = env.reset(seed=None)
    total_reward = 0
    done = False

    nn = NeuralNetwork(input_size, hidden_size, output_size, genome)

    while not done:
        output = nn.forward(obs)
        action = 1 if output[0] > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    return total_reward
