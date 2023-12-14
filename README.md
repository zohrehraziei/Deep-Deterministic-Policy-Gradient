# Deep Reinforcement Learning Repository

## Overview
This repository contains Python scripts implementing various deep reinforcement learning algorithms. It includes implementations of the Deep Deterministic Policy Gradients (DDPG) and Policy Gradient methods using TensorFlow and Keras.

## Scripts
1. `DDPG.py`: Implements the Deep Deterministic Policy Gradients Method using TensorFlow. It includes classes for replay buffer, noise, actor, and critic components essential for DDPG.
2. `PolicyGradient.py`: Contains the implementation of the Policy Gradients Method using Keras. It focuses on building a policy network and includes methods for choosing actions, storing transitions, and learning from experiences.
3. `mainddpg.py`: A script to test and run the DDPG algorithm in a gym environment, specifically the Pendulum problem.
4. `test_policyGradient.py`: Demonstrates the use of the Policy Gradient method in the LunarLander environment from the gym library.

## Requirements
- Python 3.x
- TensorFlow
- Keras
- gym
- NumPy
- Matplotlib (for plotting results)

## Usage
To use these scripts, ensure you have the required libraries installed. Then, you can run each script individually to see the implementation in action. For example, to run the DDPG algorithm on the Pendulum environment:

```bash
python mainddpg.py
