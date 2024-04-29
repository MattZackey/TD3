import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from Agent.Networks import Actor_net, Critic_net

class Agent_TD3:

    def __init__(self, env, size_memory, batch_size, gamma, tau, lr_actor, lr_critic, update_freq, policy_noise, noise_clip):
        
        self.state_dim = env.observation_space._shape[0]
        self.action_dim = env.action_space.shape[0]

        # Initialize Actor newtorks
        action_space_scale = env.action_space.high[0]
        self.actor = Actor_net(self.state_dim, self.action_dim, action_space_scale)
        self.target_actor = Actor_net(self.state_dim, self.action_dim, action_space_scale)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Initialize Critic_1 networks
        self.critic_1 = Critic_net(self.state_dim, self.action_dim)
        self.target_critic_1 = Critic_net(self.state_dim, self.action_dim)
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())

        # Initialize Critic_2 networks
        self.critic_2 = Critic_net(self.state_dim, self.action_dim)
        self.target_critic_2 = Critic_net(self.state_dim, self.action_dim)
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(size_memory))
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer_actor = optim.AdamW(self.actor.parameters(), lr = lr_actor, amsgrad=True)
        self.optimizer_critic_1 = optim.AdamW(self.critic_1.parameters(), lr = lr_critic, amsgrad=True)
        self.optimizer_critic_2 = optim.AdamW(self.critic_2.parameters(), lr = lr_critic, amsgrad=True)
        self.update_freq = update_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = env.action_space.high[0]
        

    # Add experience to memory
    ############################################################################
    def cache(self, state, action, next_state, reward, done):

        next_state = torch.tensor(next_state, dtype = torch.float32)
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state" : state, "action" : action, "next_state" : next_state,"reward" : reward, "done" : done}, batch_size=[]))
    ############################################################################

    # Update Actor and Critic
    ############################################################################
    def update(self, curr_iter):

        if len(self.memory) < self.batch_size:
            return

        #Sample a batch
        self.batch = self.memory.sample(self.batch_size)

        # Update Critics
        ####################################################

        action_batch = self.batch['action']
        with torch.no_grad():

            noise = torch.normal(mean = 0, std = self.policy_noise, size = action_batch.shape).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.target_actor(self.batch['next_state']) + noise).clamp(-self.max_action, self.max_action)

            target_Q1 = self.target_critic_1(self.batch['next_state'], next_action)
            target_Q2 = self.target_critic_2(self.batch['next_state'], next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            exp_sa = self.batch['reward'] + (target_Q * (1 - self.batch['done'].int()) * self.gamma)

        pred_sa_Q1 = self.critic_1(self.batch['state'], action_batch)
        pred_sa_Q2 = self.critic_2(self.batch['state'], action_batch)
        
        # Compute Huber loss
        Q1_loss = self.loss_fn(pred_sa_Q1, exp_sa)
        Q2_loss = self.loss_fn(pred_sa_Q2, exp_sa)

        # Reset gradients
        self.critic_1.zero_grad()
        self.critic_2.zero_grad()

        # Compute gradients
        Q1_loss.backward()
        Q2_loss.backward()

        # Update parameters
        self.optimizer_critic_1.step()
        self.optimizer_critic_2.step()
        ####################################################

        # Update Actor and target networks
        ####################################################
        if (curr_iter % self.update_freq) == 0:

            # Reset gradients
            self.optimizer_actor.zero_grad()

            actions_pred = self.actor(self.batch['state'])
            state_actions_pred = self.critic_1(self.batch['state'], actions_pred)
            actor_loss = torch.mean(-state_actions_pred)
            actor_loss.backward()

            # Update parameters
            self.optimizer_actor.step()

            # Update all target networks
            for param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        ####################################################