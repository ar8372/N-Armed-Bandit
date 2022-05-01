"""
Problem1:- 
Consider a binary bandit with two rewards {1-success, 0-failure}.  
The bandit returns 1 or 0 for the action that you select, i.e. 1 or 2.  
The rewards are stochastic (but stationary).  
Use an epsilon-greedy algorithm discussed in class and decide upon the action to take for maximizing the expected reward.  
There are two binary bandits named binaryBanditA.m and binaryBanditB.m are waiting for you.
"""
from Environment import N_Armed_Bandit
from Agent import Agents

if __name__ == "__main__":

    no_arms = 2 
    agent_name = "epsilon_greedy"
    epsilon = 0.5
    discount_factor = 0.7 # future reward importance
    random_state = 100
    _mean = 0
    _std = 1
    no_iterations = 5
    verbose = 0
    manual_reward = [1, 0] # will override random reward assignment

    print()
    # create env 
    e = N_Armed_Bandit(no_arms, random_state, _mean, _std,manual_reward = manual_reward)
    print()

    # create agent 
    a = Agents(agent_name, epsilon, discount_factor, e)


    a.train(no_iterations,verbose)
    print()
    print("True Reward is:")
    print(e.rewards)
    print("Expected Reward is:")
    print(a.expectedReward)
    print()
    print("Best action is: ",a.best_action())
    print()


