"""
Problem3:- 
Develop a 10-armed bandit in which all ten mean-rewards start out equal and 
then take independent random walks (by adding a normally distributed increment 
with mean zero and standard deviation 0.01 to all mean-rewards on each time step). 
{function [value] = bandit_nonstat(action)}

"""
from Environment import N_Armed_Bandit
from Agent import Agents

if __name__ == "__main__":

    no_arms = 10 
    agent_name = "epsilon_greedy"
    epsilon = .4 # How many times to explore
    discount_factor = 0.7 # future reward importance
    random_state = 100
    _mean = 0
    _std = 0.01
    no_iterations = 1000
    verbose = 0
    manual_reward = "--|--" #[1, 0] # will override random reward assignment

    print()
    # create env 
    e = N_Armed_Bandit(no_arms, random_state,_mean, _std, manual_reward = manual_reward)
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


