import math
import random

import numpy as np
from matplotlib import pyplot as plt

import util.util
from environment.environment import TaskAllocationEnvironment
from model.agent import Agent
from dataset import data




#agent.load_models()
np.random.seed(0)


def training():
    score = 0
    crit_value = 0
    score_history = []
    crit_history = []
    for epoch in range(30):
        print('epoch', epoch)
        for i in range(len(train)):
            game = 0
            #todo use sum of task to predict delay
            env.task = [sum(train[i])]
            env.taskList = train[i]
            obs = env.observe()
            done = False

            while not done and game < 100:
                act, crit = agent.choose_action(obs)
                # print('action', act)
                #todo maximizing q value does not maximize reward

                new_state, reward, done, _, info = env.step(act, False)
                # print(new_state)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                crit_value += crit
                obs = new_state
                game += 1

                # print('\n')
            score /= game
            crit_value /= game


                #env.render()
            score_history.append(score)
            crit_history.append(crit_value)


            #if i % 25 == 0:
            #    agent.save_models()
            print('\033[92m' + 'terminal' + str(game)+ '\033[0m')
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f ' % np.mean(score_history[-100:]))

    from util import util

    util.plotLearning(score_history, 'train', window=len(score_history))
    util.plotLearning(crit_history, 'crit', window=len(crit_history))
convergence = []
convergence_delay = []
def eval():
    score = 0
    average_delay = []
    score_history = []
    for i in range(len(test)):
        env.task = [sum(test[i])]
        env.taskList = test[i]
        obs = env.observe()
        act, _ = agent.choose_action(obs)
        # print('action', act)


        new_state, reward, done, delay, info = env.step(act, True)
        score += reward
        average_delay.append(delay/len(train[i]))
        # print(new_state)

            # env.render()
        score_history.append(score)
    from util import util
    # util.plotLearning(score_history, 'test', window=len(score_history))
    # util.plotconvergence(average_delay, score_history, 'covergence')
    print(sum(average_delay)/len(average_delay))
    print(score)
    convergence.append(score/sum(sum(test,[])))
    convergence_delay.append(sum(average_delay)/len(average_delay))
    print(sum(sum(test,[])))

from util import util
if __name__ == '__main__':
    cpu , train, test = data.getDataFromCSV('dataset/dataset10003node.csv')
    env = TaskAllocationEnvironment(cpu, 1.5)
    agent = Agent(alpha=10e-3, beta=10e-3, actor_input_dims=[1 + len(cpu)], crictic_input_dim=[1+len(cpu)], tau=0.001, env=env,
                  batch_size=64, layer1_size=10, layer2_size=300, n_actions=len(cpu))
    training()
    eval()
    agent.save_models()
