import numpy as np
from gym import Env
import random


def GenerateRandomState(n):
    rand_list = []
    for i in range(n):
        rand_list.append(random.randint(0, 1))
    return rand_list


class TaskAllocationEnvironment(Env):
    def __init__(self, cpuPower, ddl):
        self.cpuPower = cpuPower
        self.Num_worker = len(cpuPower)
        self.state = GenerateRandomState(self.Num_worker)
        self.completion = []
        self.ddl = ddl
        self.task = 30
    """
        execute action and return next state and reward
        @:param action: a list of task allocation
    """
    def step(self, action):
        info = ''
        # sort_index = np.argsort(self.taskList)
        # sort_index2 = np.argsort(action)
        self.completion = []
        for i in range(self.Num_worker):
            if self.task * action[i] / self.cpuPower[i] <= self.ddl:
                self.state[i] = 1
            else:
                self.state[i] = 0
            self.completion.append(self.task * action[i] / self.cpuPower[i])

        reward = 0
        for i in range(0, self.Num_worker, 2):
            reward += self.task * action[i] * self.state[i]

        done = len(set(self.state)) == 1
        print('reward ', reward)
        return self.state, reward, done, info

    def observe(self):
        return self.state

    """
        code from tutorial
        for environment visualization, not needed yet
    """
    def render(self):
        # Implement viz
        pass

    """
        for reset environment
    """
    def reset(self):
        pass


