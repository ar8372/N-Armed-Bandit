from Environment import N_Armed_Bandit
from Agent import Agents



if __name__ == "__main__":

    no_arms = 3 
    agent_name = "epsilon_greedy"
    epsilon = 0.2
    discount_factor = 0.5 # future reward importance
    random_state = 100
    _mean = 0
    _std = 1
    no_iterations = 500
    verbose = 0

    print()
    # create env 
    e = N_Armed_Bandit(no_arms, random_state, _mean, _std,)
    print()

    # create agent 
    a = Agents(agent_name, epsilon, discount_factor, e)

    print("="*40)
    print("For actions 3 actual reward is:")
    print(a.get_reward(3))
    print("and Agent Expected reward is:")
    print(a.expectedReward[2])
    print("="*40)

    a.train(no_iterations,verbose)

    print("="*40)
    print("For actions 3 actual reward is:")
    print(a.get_reward(3))
    print("and Agent Expected reward is:")
    print(a.expectedReward[2])
    print("="*40)

    print()
    print("Best action is: ",a.best_action())


