import torch
import torch.nn as nn
import torch.nn.functional as F
import tensordict
import torch.optim as optim
import numpy as np

from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
from Agent.Networks import Actor_net, Critic_net

class Agent_DDPG:

    def __init__(self, state_dim, action_dim, action_space_scale, size_memory, batch_size, gamma, tau, lr_actor, lr_critic):

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Initialize Actor newtorks
        self.actor = Actor_net(self.state_dim, self.action_dim, action_space_scale)
        self.target_actor = Actor_net(self.state_dim, self.action_dim, action_space_scale)
        self.target_actor.load_state_dict(self.actor.state_dict())

        # Initialize Critic networks
        self.critic = Critic_net(self.state_dim, self.action_dim)
        self.target_critic = Critic_net(self.state_dim, self.action_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.memory = TensorDictReplayBuffer(storage=LazyMemmapStorage(size_memory))
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer_actor = optim.AdamW(self.actor.parameters(), lr = lr_actor, amsgrad=True)
        self.optimizer_critic = optim.AdamW(self.critic.parameters(), lr = lr_critic, amsgrad=True)
        
    # Add experience to memory
    ############################################################################
    def cache(self, state, action, next_state, reward, done):

        next_state = torch.tensor(next_state, dtype = torch.float32)
        reward = torch.tensor([reward])
        done = torch.tensor([done])

        self.memory.add(TensorDict({"state" : state, "action" : action, "next_state" : next_state,"reward" : reward, "done" : done}, batch_size=[]))
    ############################################################################
        
    # Update all networks
    ############################################################################
    def update(self):
        
        if len(self.memory) < self.batch_size:
            return

        #Sample a batch
        self.batch = self.memory.sample(self.batch_size)

        # Update Critic
        ####################################################
        action_batch = self.batch['action']
        next_action = self.target_actor(self.batch['next_state'])
        with torch.no_grad():
            target_Q = self.target_critic(self.batch['next_state'], next_action)
        exp_sa = self.batch['reward'] + (target_Q * (1 - self.batch['done'].int()) * self.gamma)

        pred_sa = self.critic(self.batch['state'], action_batch)
        
        # Compute Huber loss
        critic_loss = self.loss_fn(pred_sa, exp_sa)

        # Reset gradients
        self.critic.zero_grad()

        # Compute gradients
        critic_loss.backward()

        # Update parameters
        self.optimizer_critic.step()
        ####################################################
            
        # Update Actor 
        ####################################################
        actions_pred = self.actor(self.batch['state'])
        state_actions_pred = self.critic(self.batch['state'], actions_pred)
        actor_loss = torch.mean(-state_actions_pred)
            
        # Reset gradients
        self.optimizer_actor.zero_grad()
            
        # Compute gradients
        actor_loss.backward()

        # Update parameters
        self.optimizer_actor.step()
        ####################################################
            
        # Update target networks
        ####################################################
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.target_actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        ############################################################################
        
        
