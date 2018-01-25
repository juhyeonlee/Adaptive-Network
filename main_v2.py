
import random
import numpy as np
import matplotlib.pyplot as plt

from env import Environment
from Qlearning import QLearningAgent
from time import time

if __name__ == '__main__':

    #plt.close('all')
    ########### tuning parameters #########
    one_dim = 7
    mu = 4 / 5
    init_txr = 2
    # random_seed = 1 #??
    epsilon = 0.4 # explore ratio
    epsilon_space_id = range(0,11)
    utility_coeff = 3 #0.95  # weight on goodput
    utility_pos_coeff = 1  # to make reward to be positive

    # action_space = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    action_space = ["%.1f" % round(i * 0.1, 1) for i in range(-10, 11)]
        #[-1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
    ep_length = 10
    ########################

    num_ep = 1
    sum_goodput = 0.
    sum_reward = 0.

    env = Environment(one_dim, mu, init_txr, utility_coeff, utility_pos_coeff, action_space)

    n_actions = env.n_actions
    action_space = env.action_space




    goodput_trace=np.zeros((10, ep_length),dtype=float)
    energy_trace=np.zeros((10, ep_length),dtype=float)
    reward_trace=np.zeros((10, ep_length),dtype=float)

    t = time() # for checking processing time
    for epsilon_id in epsilon_space_id:
        epsilon = epsilon_id * 0.1
        for episode in range(num_ep):
            # TODO: firstly set to 0, then specified particular distributions for link failure rate
            beta = 0.0 # random.uniform(0.0, 0.3)
            state = env.reset(beta, init_txr)
            agent = []
            for i in range(len(state) - 4):
                agent.append(QLearningAgent(action_space))
            action = np.zeros(len(state), dtype=np.float32)
            #goodput_trace = []
            #reward_trace = []
            #energy_trace = []
            for steps in range(ep_length):
                print('step number: ,', steps)
                for i in range(len(state) - 4):
                    action[i] = agent[i].get_action(state[i], epsilon)
                # print('action check:', action)
                next_state, reward, goodput, energy = env.step(action)

                for i in range(len(state) - 4):
                    agent[i].learn(state[i], action[i], reward[i], next_state[i])

                goodput_trace[epsilon_id, steps] = goodput
                reward_trace[epsilon_id, steps] = np.mean(reward)
                energy_trace[epsilon_id, steps] = np.sum(energy)
                state = next_state



        print('goodput trace: ', goodput_trace[epsilon_id,:])
        print('reward trace :', reward_trace[epsilon_id,:])
        print('energy trace :', energy_trace[epsilon_id,:])

    print('processing time:', time()-t)

        # for i in range(len(state)):
        #     agent[i].print_qtable()

        # test
     #   for i in range(len(state) - 4):
     #       action[i] = agent[i].get_greedy_action(state[i])
     #   next_state, reward, goodput = env.step(action)
     #   sum_goodput += goodput
     #   sum_reward += np.sum(reward)
     #   print('greedy approach - TX range: ',env.txr)



    plt.figure(0)
    legend_set = [epsilon_id*0.1 for epsilon_id in epsilon_space_id]
    plt.xlabel('episode')
    plt.ylabel('goodput')
    plt.grid()
    for epsilon_id in epsilon_space_id:
        plt.plot(range(ep_length), goodput_trace[epsilon_id,:],'-*')
    plt.legend(legend_set)
    plt.show()

    plt.figure(1)
    #plt.plot(range(ep_length), energy_trace,'+')
    plt.xlabel('episode')
    plt.ylabel('energy sum')
    plt.grid()
    for epsilon_id in epsilon_space_id:
        plt.plot(range(ep_length), energy_trace[epsilon_id, :], '-*')
    plt.legend(legend_set)
    plt.show()


    plt.figure(2)
    plt.xlabel('episode')
    plt.ylabel('reward')
    plt.grid()
    for epsilon_id in epsilon_space_id:
        plt.plot(range(ep_length), reward_trace[epsilon_id, :], '-*')
    plt.legend(legend_set)
    #plt.plot(range(ep_length), reward_trace, '+')
    plt.show()

    #print('average goodput: ', sum_goodput / num_ep, 'average reward :', sum_reward / num_ep)

