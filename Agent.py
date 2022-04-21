
import numpy as np
import random 
from utils import *

# define environment
"""
Agent:
input:=> environment, decides which actions to take
and based on the reward it get's from env it updates it's policy.
[updated idea about what is the ExpectedReward]
returns:=> Optimal policty
"""

class Agents:
    def __init__(self, name_agent, epsilon, env, n_iterations):
        #==> Types of Agent
        # Epsilon-Greedy Agent 
        # Optimistic Initial Start 
        # Upper Confidence Bound 
        # Gradient Bandit Agent
        self.name_agent = name_agent
        self.epsilon = epsilon 
        self.env = env 
        self.no_arms = self.env.no_arms
        self.actions = self.env.actions

        # initialize expectedReward 
        self.initialize_()
    
    def train(self, n_iterations, seed=2023):
        # seed it for reproducibility of result 
        seedBasis(seed)

        for iter_no in range(n_iterations):
            print(f"iteration no {iter_no}")
            action = self.take_action 
            print(f"action: {action}")
            reward = self.get_reward(action)
            print(f"reward: {reward}")

            # update
            self.update_expectedReward(action, reward)
            print(f"New reward: {self.expectedReward}")
            print()



    def initialize_(self):
        # initialize initial expected reward of each arm
        self.expectedReward = [0]*self.no_arms

    def take_actions(self):
        # returns argmax actions 1-epsilon times 
        prob = random.uniform(0,1)
        if prob > 1- self.epsilon:
            # argmax 
            action = np.argmax(self.expectedReward)
        else:
            # random choice 
            action = np.random.choice(self.actions)
        return action

    def get_reward(self, action):
        # environment will provide reward 
        reward_received = self.env.reward(action)




    def update_expectedReward(self, action_taken, reward_received):
        pass 
        