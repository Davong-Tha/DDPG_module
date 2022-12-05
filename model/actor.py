import os

import numpy as np

import torch
import torch.nn as nn
import torch as T
from torch import optim
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='tmp/ddpg'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.checkpoint_file = os.path.join(chkpt_dir, name + '_ddpg')


        self.fc1 = nn.Linear(*self.input_dims, self.n_actions)
        f1 = 1. / np.sqrt(self.fc1.weight.data.size()[0])
        T.nn.init.uniform_(self.fc1.weight.data, -f1, f1)
        T.nn.init.uniform_(self.fc1.bias.data, -f1, f1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.3)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """

        :param state:

        :return: load capacity for each worker
        """
        x = self.fc1(state)
        if(len(state.shape) == 1):
            x = torch.div(state[0], x)
        else:
            temp = torch.unsqueeze(state[:, 0], dim=1)
            x = torch.mul(x, 1/temp)
        x = x * 5
        # x = F.relu(x)
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

