import os

import numpy as np
import torch

import torch.nn as nn
import torch as T
import torch.nn.functional as F
from torch import optim

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')


        self.fc1 = nn.Linear(3, self.fc1_dims)

        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        # self.fc1.weight.data.uniform_(-f1, f1)
        # self.fc1.bias.data.uniform_(-f1, f1)
        self.bn1 = nn.LayerNorm(self.fc1_dims)


        self.state_value = nn.Linear(3, self.fc1_dims)
        temp = 1. / np.sqrt(self.state_value.weight.data.size()[0])
        T.nn.init.uniform_(self.state_value.weight.data, -temp, temp)
        T.nn.init.uniform_(self.state_value.bias.data, -temp, temp)

        self.fc2 = nn.Linear(self.fc1_dims, 1)

        f2 = 1. / np.sqrt(self.fc2.weight.data.size()[0])
        # f2 = 0.002
        T.nn.init.uniform_(self.fc2.weight.data, -f2, f2)
        T.nn.init.uniform_(self.fc2.bias.data, -f2, f2)


        self.optimizer = optim.Adam(self.parameters(), lr=beta, weight_decay=0.3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)


    def forward(self, action, state):
        x = self.fc1(action)


        if (state.shape[0] == 4):
            state_value = self.state_value(state[1:])
        else:
            temp = state[:, 1:]
            state_value = self.state_value(temp)
        x = self.fc2(T.add(x, state_value))
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

