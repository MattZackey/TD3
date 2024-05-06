import gymnasium as gym
import imageio
import pathlib
import os
import torch

def record_agent(agent_env, agent, num_iter, example_path, random_agent = False, seed = None):

    if not os.path.exists(example_path):
        pathlib.Path(example_path).mkdir(parents=True, exist_ok=True)

    env = gym.make(agent_env, render_mode = "rgb_array")
    agent.actor.eval()
    
    if seed:
        state, info = env.reset(seed = seed)
    else:
        state, info = env.reset()
    
    state = torch.tensor(state, dtype = torch.float32)
    done = False
    frames = []
    total_reward = 0
    while not done:
        frames.append(env.render())
        
        # Random agent
        if random_agent:
            action = env.action_space.sample()
            next_state, reward, terminated, truncated, _ = env.step(action)
        
        # Trained agent
        else:    
            action = agent.actor(state.view(1, -1))[0]
            next_state, reward, terminated, truncated, _ = env.step(action.tolist())
        
        done = terminated or truncated
        state = torch.tensor(next_state, dtype = torch.float32)
        total_reward += reward
    env.close()
    
    imageio.mimsave(os.path.join(example_path, f'run{num_iter}.gif'), frames, fps=30)
    
    return total_reward