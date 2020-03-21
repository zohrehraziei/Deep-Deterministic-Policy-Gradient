# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 17:43:46 2020
Deep Deterministic Policy Gradients Method

@author: raziei.z@husky.neu.edu
"""
# need replay buffer class
# use replay buffer to address the issue between samples generated on subsequent steps within an episode
# need class for target Q network  (a function of state and action) 
# we will use batch norm
# the policy is deterministic, so how to handle explore-exploit delimma? 
# answer: use stochastic policy to solve deterministic policy
# deterministic policy means to output the actual action instead of probability
# we need a way to bound action to env limit
# we have two actors and two critics networsks, a target for each
# has four NNs, two on-policy and two off policy
# update are soft for parameter of the two target networks - theta_prime = tau*theta + (1-tau)*theta_prime with tau<<1
# the target actor is just evaluation acotr plus some noise process (N)
# they use Ornstein Uhlenbeck -> need a class for noise
# so need claas for replay buffer, batch normalization, noise
