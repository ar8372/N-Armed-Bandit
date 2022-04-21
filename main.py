from Environment import N_Armed_Bandit
from Agent import Agents



if __name__ == "__main__":

    no_arms = 10 
    agent_name = "epsilon_greedy"
    epsilon = 0.2
    random_state = 100

    print()
    # create env 
    e = N_Armed_Bandit(no_arms, random_state)
    print()

    # create agent 
    a = Agents(agent_name, epsilon, e)

    a.get_reward(3)

    a.train(2000,1)

    print(e.rewards)





    print()


