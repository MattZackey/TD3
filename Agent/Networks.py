import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor neural network
#################################################################################
class Actor_net(nn.Module):

    def __init__(self, state_dim, action_dim, action_space_scale):
        super(Actor_net, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_space_scale = action_space_scale

        self.layer1 = nn.Linear(self.state_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, self.action_dim)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        x = F.tanh(self.layer3(x))
        x = x * self.action_space_scale
        return x
#################################################################################

# Critic neural network
#################################################################################
class Critic_net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Critic_net, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.layer1 = nn.Linear(self.state_dim + self.action_dim, 400)
        self.layer2 = nn.Linear(400, 300)
        self.layer3 = nn.Linear(300, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], 1)

        x = self.layer1(x)
        x = F.relu(x)

        x = self.layer2(x)
        x = F.relu(x)

        x = self.layer3(x)

        return x
#################################################################################