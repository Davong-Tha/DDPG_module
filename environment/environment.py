import numpy as np
from gym import Env
import random

from BnB.branchNbound import GetBnBAllocation


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
    def step(self, action, debug, show_delay=False):
        info = ''


        # sort_index = np.argsort(self.taskList)
        # sort_index2 = np.argsort(action)

        # allocation = np.array(self.allocateTask(sorted(self.taskList), action))
        allocation = self.allocate_task_bnb(action)
        delay = float(np.max(allocation / self.cpuPower))
        self.state = allocation / self.cpuPower <= self.ddl
        reward = self.state * allocation
        # cvx.sum(cvx.maximum((cvx.multiply(task_sizes @ allocation_matrix, 1 / cpu_powers) - overall_deadlines), 0))
        temp_delay = (allocation / self.cpuPower) - self.ddl
        temp_objective  = np.sum(np.where(temp_delay < 0, 0, temp_delay))


        done = all(d == 1 for d in self.state)
        if debug:
            print(self.taskList)
            print('action', action)
            print('allocation', allocation)
            print('reward ', reward)
            print('state', self.state)
            print('\n')

        return self.task + list(self.state), float(np.sum(reward)), done, delay, info, temp_objective

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


    # def get_cpu_power_from_predicted_capacity(self):

    def allocate_task_bnb(self, cpu_powers):
        opt_ass, _ = GetBnBAllocation(self.taskList, cpu_powers, self.ddl)
        return self.taskList @ opt_ass



