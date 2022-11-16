from ddpg_torch import Agent
import gym
import numpy as np
from environment import TaskAllocationEnvironment


env = TaskAllocationEnvironment([1,2,3], 5)
agent = Agent(alpha=0.000025, beta=0.00025, input_dims=[3], tau=0.001, env=env,
              batch_size=64,  layer1_size=400, layer2_size=300, n_actions=3)

#agent.load_models()
np.random.seed(0)

score_history = []
for i in range(1000):
    obs = env.observe()
    done = False
    score = 0
    while not done:
        act = agent.choose_action(obs)
        print('observation: ', obs)
        print('action', act)
        #todo need a way to link action and dataset(the cpu power ranking is working but is not generalize)

        new_state, reward, done, info = env.step(act)
        print(new_state)
        agent.remember(obs, act, reward, new_state, int(done))
        agent.learn()
        score += reward
        obs = new_state
        #env.render()
    score_history.append(score)

    #if i % 25 == 0:
    #    agent.save_models()

    print('episode ', i, 'score %.2f' % score,
          'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))


