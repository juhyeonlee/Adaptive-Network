
import random
import numpy as np
#import matplotlib.pyplot as plt

from env import Environment
from Qlearning import QLearningAgent

if __name__ == '__main__':

    # init parameters
    one_dim = 7
    mu = 4 / 5
    init_txr = 3
    # random_seed = 1 #??

    env = Environment(one_dim, mu, init_txr)

    n_actions = env.n_actions
    action_space = env.action_space


    ep_length = 10
    num_ep = 1
    sum_goodput = 0.
    sum_reward = 0.


    for episode in range(num_ep):
        # TODO: firstly set to 0, then specified particular distributions for link failure rate
        beta = 0.0 # random.uniform(0.0, 0.3)
        state = env.reset(beta, init_txr)
        agent = []
        for i in range(len(state) - 4):
            agent.append(QLearningAgent(action_space))
        action = np.zeros(len(state), dtype=np.int32)
        goodput_trace = []
        reward_trace = []
        for steps in range(ep_length):
            for i in range(len(state) - 4):
                action[i] = agent[i].get_action_idx(state[i])
                #action[i] = agent[i].get_action(state[i])
            next_state, reward, goodput = env.step(action)

            for i in range(len(state) - 4):
                agent[i].learn(state[i], action[i], reward[i], next_state[i])

            goodput_trace.append(goodput)
            reward_trace.append(np.mean(reward))
            state = next_state

        #plt.plot(range(ep_length), goodput_trace)
        #plt.show()
        #plt.plot(range(ep_length), reward_trace)
        #plt.show()

        # for i in range(len(state)):
        #     agent[i].print_qtable()
        # test
        for i in range(len(state) - 4):
            action[i] = agent[i].get_greedy_action(state[i])
        next_state, reward, goodput = env.step(action)
        sum_goodput += goodput
        sum_reward += np.sum(reward)
        print('greedy approach - TX range: ',env.txr)

    print('goodput trace: ', goodput_trace)
    print('reward trace :', reward_trace)
    #print('average goodput: ', sum_goodput / num_ep, 'average reward :', sum_reward / num_ep)

