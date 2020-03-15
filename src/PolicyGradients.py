"""
Created on Sat Mar 14 18:27:59 2020

Policy Gradients Method

@author: raziei.z@husky.neu.edu
"""
from keras.layers import Dense, Activation, Input
from keras.models import Model, load_model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np

#### Initialize
class Agent(object):
    def __init__(self, ALPHA, GAMMA=0.99, n_actions=4,
                 layer1_size=16, layer2_size=16, input_dims=128,
                 fname='reinforce.h5'):
        self.gamma = GAMMA
        self.lr = ALPHA
        self.G = 0 #discounted sum of reward at each time step
        self.input_dims = input_dims
        self.fc1_dims = layer1_size
        self.fc2_dims = layer2_size
        self.n_actions = n_actions
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        #set single set episode as our batch (we could do a batch of episode)
        self.policy, self.predict = self.build_policy_network()
        self.action_space = [i for 1 in range(n_actions)]
        self.model_file = fname
        
#### Build Policy Network Function
    def build_policy_network(self):
       input= Input(shape=(self.input_dims,))
       dense1 = Dense(self.fc1_dims, activation ='relu')(input)
       dense2 = Dense(self.fc2_dims, activation= 'relu')(dense1)
       probs = Dense(self.n_actions, activation= 'softmax')(dense2)
       
 ######Loss Function: Keras do not have losss function and we need to write it our own

       def custom_loss(y_true, y_pred):
           out = K.clip(y_pred, 1e-8, 1-1e-8)
           log_lik = y_true*K.log(out)
           
           return K.sum(-log_lik*advantages)
###advantages is tell us reward of taking paticular action at each time step
           
       policy = Model(input=[input, advantages], output=[probs])
       policy.compile(optimizer=Adam(lr=self.lr), loss=custom_loss)
       
       predict = Model(input=[input], output=[probs])
       
       return policy, predict
   
    def choose_action(self, observation):
        state = observation[np.newaxis, :]
        probabilities = self.predict.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)
        
        return action
    
    def store_transition(self, observation, action, reward):
        self.action_memory.append(action)
        self.state_memory.append(observation)
        self.reward_memory.append(reward)
        
        
    def learn(self):
         h
        
        
        

       
        
       
        
           
        
    