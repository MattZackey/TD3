import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
import random
import imageio
import pathlib
import os
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from itertools import count
from Agent.TD3 import Agent_TD3

# Setup game
env = gym.make('HalfCheetah-v4')

# Set random number seeds
seed_value = 9809
torch.manual_seed(seed_value)
np.random.seed(seed_value)   
random.seed(seed_value)      

# Setup agent
agent_cheetah = Agent_TD3(env = env,
                          size_memory = 1000000,
                          batch_size = 100,
                          gamma = 0.99,
                          tau = 0.005,
                          lr_actor = 0.001,
                          lr_critic = 0.001,
                          update_freq = 2,
                          policy_noise = 0.2,
                          noise_clip = 0.5)

# Train agent
score_train = []
num_episodes = 10
num_random_samples = 5000
reward_eval = []
for i_episode in range(num_episodes):
    
    state,_ = env.reset()
    state = torch.tensor(state, dtype = torch.float32)
    total_reward = 0

    for t in count():

        if num_random_samples > 0:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype = torch.float32)
            num_random_samples -= 1
        
        else:
            with torch.no_grad():
                action = agent_cheetah.actor(state.view(1, -1))[0]
            action += torch.normal(mean = 0, std = 0.1, size = action.shape)
        
        next_state, reward, terminated, truncated, _ = env.step(action.tolist())
        done = terminated or truncated

        total_reward += reward

        # Store transition pair
        agent_cheetah.cache(state, action, next_state, reward, done)

        # Update critic and actor
        agent_cheetah.update(i_episode)

        # Update state
        next_state = torch.tensor(next_state, dtype = torch.float32)
        state = next_state

        if done:
            score_train.append(total_reward)
            break
