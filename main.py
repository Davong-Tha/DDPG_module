import random

import numpy as np
from matplotlib import pyplot as plt

from environment.environment import TaskAllocationEnvironment
from model.agent import Agent
from dataset import data

env = TaskAllocationEnvironment([10, 20, 30], 5)
agent = Agent(alpha=10e-3, beta=10e-3, input_dims=[1], tau=0.001, env=env,
              batch_size=64,  layer1_size=200, layer2_size=300, n_actions=3)

#agent.load_models()
np.random.seed(0)


def training():
    score = 0
    score_history = []
    for epoch in range(5):
        print('epoch', epoch)
        for i in range(len(train)):
            game = 0
            #todo use sum of task to predict delay
            env.task = [sum(train[i])]
            env.taskList = train[i]
            obs = env.observe()
            done = False

            while not done and game < 100:
                act = agent.choose_action(obs)
                print('action', act)
                #todo maximizing q value does not maximize reward

                new_state, reward, done, info = env.step(act)
                print(new_state)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
                game += 1
                print('\n')


                #env.render()
            score_history.append(score)

            #if i % 25 == 0:
            #    agent.save_models()
            print('\033[92m' + 'terminal' + str(game)+ '\033[0m')
            print('episode ', i, 'score %.2f' % score,
                  'trailing 100 games avg %.3f ' % np.mean(score_history[-100:]))

    from util import util

    util.plotLearning(score_history, 'train', window=len(score_history))

def eval():
    score = 0
    score_history = []
    for i in range(len(test)):
        env.task = [sum(test[i])]
        env.taskList = test[i]
        obs = env.observe()
        act = agent.choose_action(obs)


        new_state, reward, done, info = env.step(act)
        score += reward
        print(new_state)

            # env.render()
        score_history.append(score)
    from util import util
    util.plotLearning(score_history, 'test', window=5)
    print(score)
    print(sum(sum(test,[])))


if __name__ == '__main__':
    train, test = data.getDataList('dataset/100dataset.txt')
    training()
    eval()


