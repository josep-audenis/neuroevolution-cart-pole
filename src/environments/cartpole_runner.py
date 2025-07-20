import gymnasium as gym
import numpy as np
import imageio.v2 as imageio
from evolution.neural_network import NeuralNetwork

def evaluate_genome(genome, input_size, hidden_size, output_size, n_genome, generation,render=False):
    env = gym.make("CartPole-v1", render_mode="rgb_array" if render else None)
    obs, _ = env.reset(seed=None)
    total_reward = 0
    done = False
    frames = []

    nn = NeuralNetwork(input_size, hidden_size, output_size, genome)

    while not done:
        if render:
            frame = env.render()
            frames.append(frame)

        output = nn.forward(obs)
        action = 1 if output[0] > 0 else 0
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    env.close()
    
    if render and (total_reward > 499 or (total_reward > 100 and total_reward < 150)):
        imageio.mimsave(f"assets/gifs/cartpole_{generation}.{n_genome}_{total_reward}.gif", frames, fps=45, loop=0)
        # print(f"Saved GIF to assets/cartpole_{generation}.{n_genome}_{total_reward}.gif")

    return total_reward
