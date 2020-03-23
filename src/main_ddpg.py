import os
import gym
import numpy as np
from DDPG import Agent
from utils_DDGP import plotLearning



if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    agent = Agent(alpha=0.0001, beta=0.001, input_dims=[3], tau=0.001,
                  env=env, batch_size=64, layer1_size=400, layer2_size=300,
                  n_actions=1)
    np.random.seed(0)
    score_history = []
    for i in range(1000):
        obs = env.reset()
        done = False
        score = 0
        while not done:
            act = agent.choose_action(obs)
            new_state, reward, done, info = env.step(act)
            agent.remember(obs, act, reward, new_state, int(done))
            agent.learn()
            score += reward
            obs = new_state
            #env.render()
        score_history.append(score)
        print('episode ', i, 'score %.2f' % score,
              '100 games avgerae %.3f' % np.mean(score_history[-100:]))

    filename = 'Pendulum-alpha00005-beta0005-800-600-optimized.png'
    plotLearning(score_history, filename, window=100)