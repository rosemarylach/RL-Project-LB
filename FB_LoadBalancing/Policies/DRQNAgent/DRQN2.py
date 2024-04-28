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

        self.hidden_space = 32
        self.state_space = state_space
        self.action_space = action_space

        self.Linear1 = nn.Linear(self.state_space, self.hidden_space)
        self.Linear2 = nn.Linear(self.state_space, self.hidden_space)
        self.lstm = nn.LSTM(2*self.hidden_space, self.hidden_space, batch_first=False)
        self.Linear3 = nn.Linear(self.hidden_space, self.action_space)

    def forward(self, x, h, c):

        batch_size = x.shape[1]
        x_sinr = F.relu(self.Linear1(x[..., 0]))
        x_association = F.relu(self.Linear2(x[..., 1]))

        x = torch.cat([x_sinr, x_association], dim=-1)
        x, (new_h, new_c) = self.lstm(x, (h, c))
        x = x.reshape(x.shape[0], batch_size, x.shape[2])
        x = self.Linear3(x)

        return x, new_h, new_c

    def sample_q_value(self, obs, h, c, epsilon):

        output = self.forward(obs, h, c)

        out = output[0].squeeze(0).squeeze(0).detach().numpy()
        if random.random() < epsilon:
            return np.random.rand(out.shape[0], out.shape[1]), output[1], output[2]
        else:
            return out, output[1], output[2]

    def init_hidden_state(self, batch_size):

        return torch.zeros([1, batch_size, self.hidden_space]), torch.zeros([1, batch_size, self.hidden_space])


