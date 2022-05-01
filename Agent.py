
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
    def __init__(self, name_agent, epsilon, discount_factor, env):
        #==> Types of Agent
        # Epsilon-Greedy Agent 
        # Optimistic Initial Start 
        # Upper Confidence Bound 
        # Gradient Bandit Agent
        self.name_agent = name_agent
        self.epsilon = epsilon 
        self.discount_factor = discount_factor
        self.env = env 
        self.no_arms = self.env.no_arms
        self.actions = self.env.actions

        # initialize expectedReward 
        self.initialize_()
    
    def train(self, n_iterations, verbose=0, seed=2023):
        # seed it for reproducibility of result 
        seedBasic(self,seed)

        for iter_no in range(n_iterations):
            action = self.take_action() 
            reward = self.get_reward(action)
            if verbose >= 1:
                print(f"iteration no {iter_no}")
                print(f"action: {action}")
                print(f"reward: {reward}")
                print()

            # update
            self.update_expectedReward(action, reward)
            print(f"iter no {iter_no}: {self.expectedReward}")



    def initialize_(self):
        # initialize initial expected reward of each arm
        self.expectedReward = [0]*self.no_arms
        print(f"Initial expectedRewards: ")
        print(self.expectedReward)
        print()

    def best_action(self):
        action = self.actions[np.argmax(self.expectedReward)]
        return action

    def take_action(self):
        # 1-epsilon times exploit
        # so epsilon tells how many times explore 
        prob = random.uniform(0,1) #[low,high)
        if prob > self.epsilon:
            # argmax 
            action = self.actions[np.argmax(self.expectedReward)]
        else:
            # random choice 
            action = np.random.choice(self.actions)
        return action

    def get_reward(self, action):
        # environment will provide reward 
        reward_received = self.env.reward(action)
        return reward_received


    def update_expectedReward(self, action_taken, reward_received):
        index_ = self.actions.index(action_taken)
        self.expectedReward[index_] += self.discount_factor*(reward_received -self.expectedReward[index_] ) 

        