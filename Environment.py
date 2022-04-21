
import numpy as np 
import random

from utils import *

# define environment

"""
Environment: [Stationary Environment but non-deterministic]
input:=> action
returns:=> reward
"""
class N_Armed_Bandit:
    def __init__(self,n, random_state):
        self.no_arms = n
        self.actions = [i for i in range(1, self.no_arms+1)]
        print(f"Actions: ")
        print(self.actions)
        
        # seed everything 
        seedBasic(self,random_state)

        # initialize 
        self.initialize_()

    def initialize_(self):
        # initialize reward of each arm actual reward associated with each action
        self.rewards = [np.random.randint(1,100) for i in range(self.no_arms)]
        print(f"Means of Reward Distributions")
        print(self.rewards)

    def reward(self, action):
        # returns reward on taking a given action 
        prob_dist = random.gauss(0,1)
        return self.rewards[self.actions.index(action)] + prob_dist



