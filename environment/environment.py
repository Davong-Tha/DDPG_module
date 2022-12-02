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
        self.cpuPower = np.array(cpuPower)
        self.Num_worker = len(cpuPower)
        self.state = GenerateRandomState(self.Num_worker)
        self.completion = [1,1,1]
        self.ddl = ddl
        self.task = []
        self.taskList = []
    """
        execute action and return next state and reward
        @:param action: a list of task allocation
    """
    def step(self, action):
        info = ''
        # sort_index = np.argsort(self.taskList)
        # sort_index2 = np.argsort(action)

        allocation = np.array(self.allocateTask(sorted(self.taskList), action))
        delay = float(np.sum(allocation / self.cpuPower))
        self.state = allocation / self.cpuPower < self.ddl
        reward = self.state * allocation

        done = all(d == 1 for d in self.state)
        # print('allocation', allocation)
        # print('reward ', reward)
        # print('state', self.state)
        return self.task + list(self.state), float(np.sum(reward)), done, delay, info

    def observe(self):
        return self.task + list(self.state)

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

    def allocateTask(self, sort_train, predicted_load_capacity):
        allocation = [0] * predicted_load_capacity

        for a in sort_train:
            assign = False
            for i in range(len(predicted_load_capacity)):
                if a < predicted_load_capacity[i] - allocation[i]:
                    allocation[i] += a
                    assign = True
                    break

            if assign:
                continue

            exceed = []
            for i in range(len(predicted_load_capacity)):
                exceed.append(a - predicted_load_capacity[i] + allocation[i])
            best_exceed = np.argmin(np.abs(np.array(exceed)))
            allocation[best_exceed] += a
        return allocation


