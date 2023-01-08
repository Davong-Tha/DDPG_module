import math
import random

import numpy as np
from matplotlib import pyplot as plt

import util.util
from BnB.branchNbound import GetBnBAllocation
from environment.environment import TaskAllocationEnvironment
from model.agent import Agent
from dataset import data
from HungarianMethod.HungarianMethod import HungarianMethod
from tqdm import tqdm




#agent.load_models()
np.random.seed(0)


# def setDeadLine(data):
#     optimal_allocation = h.allocate(data, len(env.cpuPower), env.cpuPower)
#     max_ddl = 0
#     for index, sol in enumerate(optimal_allocation):
#         if len(sol) > 0:
#             tasks = []
#             for i in sol:
#                 tasks.append(data[i])
#             ddl = sum(tasks) / env.cpuPower[index]
#             if ddl > max_ddl:
#                 max_ddl = ddl
#     env.ddl = max_ddl
#     return optimal_allocation, env.ddl

def setDeadLine(data):
    ddl = np.mean(data)/np.mean(env.cpuPower)
    optimalAllocation, _  = GetBnBAllocation(data, env.cpuPower, ddl)
    # max_dll = 0
    # #find each ddl
    # for i in range(len(env.cpuPower)):
    #     task = 0
    #     for idx, j in enumerate(optimalAllocation[i]):
    #         if j == 1:
    #             task += data[idx]
    #     ddpgDDL = task/env.cpuPower[i]
    #     if ddpgDDL > max_dll:
    #         max_dll = ddpgDDL
    env.ddl = ddl
    # return max_dll
    return ddl

def get_optimal_objective(task_sizes):
    optimal_assignment, optimal_objective = GetBnBAllocation(task_sizes, env.cpuPower, env.ddl)

    return optimal_objective

def training():
    score = 0
    crit_value = 0
    score_history = []
    crit_history = []
    state_history = []
    temp_objectives = []
    optimal_objectives = []

    for epoch in range(1):
        # print('epoch', epoch)
        for i in range(len(train) - 700):
            game = 0
            #todo use sum of task to predict delay
            if (len(train[i]) == 1):
                continue
            env.task = [sum(train[i])]
            ddl = setDeadLine(train[i])


            env.taskList = train[i]
            obs = env.observe()
            done = False

            while not done and game < 100:
                act, crit = agent.choose_action(obs, ddl)
                # print('action', act)
                #todo maximizing q value does not maximize reward

                new_state, reward, done, _, info, temp_objective = env.step(act, True)

                # print(new_state)
                agent.remember(obs, act, reward, new_state, int(done), ddl)
                agent.learn()
                score += reward
                crit_value += crit
                obs = new_state
                game += 1

            temp_objectives.append(temp_objective)
            optimal_objectives.append(get_optimal_objective(train[i]))
            state_history.append(np.sum(new_state[1:]))
                # print('\n')
            score /= game
            crit_value /= game


                #env.render()
            score_history.append(score)
            crit_history.append(crit_value)
            print('\033[92m' + 'terminal' + str(game) + '\033[0m')
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f ' % np.mean(score_history[-100:]))
        eval()



    # plt.plot(state_history)
    plt.plot(temp_objectives, label="temp objective")
    plt.plot(optimal_objectives, label="optimal objective")
    plt.legend()
    plt.show()


            # if i % 25 == 0:
            #    agent.save_models()


    from util import util

    util.plotLearning(score_history, 'train', window=len(score_history))
    util.plotLearning(crit_history, 'crit', window=len(crit_history))
convergence = []
convergence_delay = []
def eval():
    score = 0
    max_delay = []
    score_history = []
    for i in range(len(test)):
        env.task = [sum(test[i])]
        env.taskList = test[i]
        ddl = setDeadLine(test[i])
        # print(setDeadLine(test[i]))
        obs = env.observe()
        act, _ = agent.choose_action(obs, ddl)
        # print('action', act)


        new_state, reward, done, delay, info, _ = env.step(act, True, False)
        score += reward
        max_delay.append(delay)

        # print(new_state)

            # env.render()
        score_history.append(score)
    from util import util
    # util.plotLearning(score_history, 'test', window=len(score_history))
    # util.plotconvergence(average_delay, score_history, 'covergence')
    # print(sum(average_delay)/len(average_delay))
    print(sum(max_delay)/len(max_delay))
    print(score)
    convergence.append(score/sum(sum(test,[])))
    # convergence_delay.append(sum(average_delay)/len(average_delay))
    print(sum(sum(test,[])))

from util import util
if __name__ == '__main__':
    cpu , train, test = data.getDataFromCSV('dataset/dataset10003node.csv')
    env = TaskAllocationEnvironment(cpu, 1.5)
    agent = Agent(alpha=10e-3, beta=10e-3, actor_input_dims=[1 + len(cpu)], crictic_input_dim=[1+len(cpu)], tau=0.001, env=env,
                  batch_size=64, layer1_size=10, layer2_size=300, n_actions=len(cpu))
    h = HungarianMethod()
    training()
    agent.save_models()
