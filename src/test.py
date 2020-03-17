# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 09:06:20 2020

@author: zohre
"""

import gym
import matplotlib.pyplot as plt
import numpy as np
from PolicyGradients import Agent
from utils import plotLearning
#from gym.envs.registration import registry, register, make, spec
#register(
#    id='CartPole-v1',
#    entry_point='gym.envs.classic_control:AcrobotEnv',
#    reward_threshold=-100.0,
#    max_episode_steps=500,
#)

if __name__ == '__main__':
    agent = Agent(ALPHA=0.0005, input_dims=8 , GAMMA=0.99, n_actions=4,
                 layer1_size=64, layer2_size=64)
    env = gym.make('CartPole-v1')
    score_history = []
    
    n_episodes = 2000
    
    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)    
        
        agent.learn()
        
        print('episode', i, 'score %.lf' % score,
               'average_score %.lf' % np.mean(score_history[-100:]))
        
    filename = 'lunar_lander.png'
    plotLearning(score_history, filename=filename, window=100)        
            
            