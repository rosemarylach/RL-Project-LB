import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Q_network
class Q_net(nn.Module):
    def __init__(self, state_space=None,
                 action_space=None):
        super(Q_net, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be selected."
        assert action_space is not None, "None action_space input: action_space should be selected."

        self.hidden_space = 64
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear3 = nn.Linear(2 * self.hidden_space, self.action_space)

    def forward(self, x):
        x_sinr = F.relu(self.Linear1(x[..., 0]))
        x_association = F.relu(self.Linear2(x[..., 1]))

        x = torch.cat([x_sinr, x_association], dim=-1)
        x = self.Linear3(x)

        return x

    def sample_q_value(self, obs, epsilon):

        output = self.forward(obs)

        out = output.squeeze().detach().numpy()
        if random.random() < epsilon:
            return np.random.rand(out.shape[0], out.shape[1])
        else:
            return out
