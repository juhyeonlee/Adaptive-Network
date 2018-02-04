
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import argparse


from env import Environment
from DQNAgent import DQNAgent


if __name__ == '__main__':

    # experiment
    # one_dim_set = [6, 8, 10, 20, 30]
    # utility_coeff_set = [1.0, 3.0, 5.0, 7.0]
    # beta_set = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--nw_size", help="network size", type=int, default=7)
    parser.add_argument("--coeff", help="utility coeff", type=float, default=0.8)
    parser.add_argument("--beta", help="beta, link failure rate", type=float, default=0.0)
    args = parser.parse_args()

    save_goodput = []
    save_reward = []
    save_energy = []
    save_connect_ratio = []
    save_txr = []
    save_num_players = []

    if not os.path.exists('./model'):
        os.makedirs('./model')

    ########### tuning parameters #########
    one_dim = args.nw_size
    mu = 4 / 5
    init_txr = 0
    # random_seed = 1 #??

    # utility coefficient
    utility_coeff = args.coeff  # weight on goodput
    utility_pos_coeff = 5  # to make reward to be positive --> theoretically, set it as zero since negative utility is OK --> but experimentally, that's wrong

    # action_space = [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1]
    action_space = ["%.1f" % round(i * 0.1, 1) for i in range(-10, 11)]
    # action_space = ["%.1f" % round(i * 0.25, 1) for i in range(0, 21)]
    #[-1.00, -0.90, -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20, -0.10, 0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]


    #if UCB is used, epsilon is meaningless
    epsilon = {'epsilon_start': 1.0, 'epsilon_end': 0.01, 'epsilon_step': 100}
    # 왜인지 모르겠지만, epsilon_step 이 100을 넘어가면 dqn이 energy를 과도하게 줄이는 방향으로 설정됨
    #epsilon = {'epsilon_start': 0.1, 'epsilon_end': 0.1, 'epsilon_step': 100}
    ep_length = 200#00
    num_ep = 1#000

    learning_rate = 0.01
    discount_factor = 0.7

    beta = args.beta
    ########################

    env = Environment(one_dim, mu, init_txr, utility_coeff, utility_pos_coeff, action_space)

    n_actions = env.n_actions
    action_space = env.action_space


    for episode in range(num_ep):
        state, num_players = env.reset(init_txr, beta) # len(state) =  num_players + 4
        agent = []
        tf.reset_default_graph()
        sess = tf.Session()
        for i in range(len(state) - 4):
            agent.append(DQNAgent(sess, 1, action_space, discount_factor, i, epsilon, learning_rate))
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        action = np.zeros(len(state), dtype=np.float32)
        q_value = np.zeros((len(state), n_actions), dtype=np.float32)
        goodput_trace = []
        reward_trace = []
        energy_trace = []
        con_ratio_trace = []
        txr_trace = []
        qvalue_trace = []
        for steps in range(ep_length):
            print('step number: ,', steps)
            #beta_idx = steps//(ep_length/len(beta_set))
            #beta = beta_set[int(beta_idx)]
            for i in range(len(state) - 4):
                action[i], q_value[i] = agent[i].get_action(state[i])
            # print('action check:', action)
            next_state, reward, goodput, energy, con_ratio = env.step(action, steps)

            #print('total energy sum: ', np.sum(energy))
            #print('number of agemts: ',num_players )
            #print('energy per an agent: ',np.sum(energy)/num_players )

            for i in range(len(state) - 4):
                agent[i].learn(state[i], action[i], reward[i], next_state[i])

            state = next_state

            goodput_trace.append(goodput)
            reward_trace.append(np.mean(reward))
            energy_trace.append(np.sum(energy)/num_players) #energy per an agent
            #print('energy trace: ', energy_trace)
            con_ratio_trace.append(con_ratio)
            txr_trace.append(env.txr)
            qvalue_trace.append(np.mean(q_value))


        # test
        for i in range(len(state) - 4):
            action[i] = agent[i].get_greedy_action(state[i])
        next_state, reward, goodput, energy, con_ratio = env.step(action, ep_length)
        goodput_trace.append(goodput)
        reward_trace.append(np.mean(reward))
        energy_trace.append(np.sum(energy)/num_players) #energy per an agent
        con_ratio_trace.append(con_ratio)
        txr_trace.append(env.txr)



        save_goodput.append(goodput_trace)
        save_reward.append(reward_trace)
        save_energy.append(energy_trace)
        save_connect_ratio.append(con_ratio_trace)
        save_txr.append(txr_trace)
        save_num_players.append(num_players)
        saver.save(sess, './model/model_ep'+ str(episode) + '_nw' + str(one_dim) + '_beta' + str(beta) + '_coeff' + str(utility_coeff) + '.ckpt')

        # goodput_trace.append(goodput)
        # reward_trace.append(np.mean(reward))
        # energy_trace.append(np.sum(energy))
        # print('greedy approach - TX range: ', env.txr)

        # print('goodput trace: ', goodput_trace)
        # print('reward trace :', reward_trace)

        plt.figure(0)
        plt.plot(range(ep_length+1), goodput_trace,'-*')
        plt.xlabel('episode')
        plt.ylabel('goodput')
        #plt.show()

        plt.figure(1)
        plt.plot(range(ep_length+1), energy_trace,'-+')
        plt.xlabel('episode')
        plt.ylabel('energy per an agent')


        plt.figure(2)
        plt.plot(range(ep_length+1), con_ratio_trace,'-+')
        plt.xlabel('episode')
        plt.ylabel('connectivity ratio')

        plt.show()
        #
        #
        plt.figure(2)
        plt.plot(range(ep_length), qvalue_trace,'-+')
        plt.xlabel('episode')
        plt.ylabel('Q value')
        plt.show()

    # Saving the objects:
    with open('var_nw' + str(one_dim) + '_beta' + str(beta) + '_coeff' + str(utility_coeff) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([save_reward, save_goodput, save_connect_ratio, save_energy, save_txr], f)

    # print('average goodput: ', np.sum(sum_goodput), 'average reward :', np.sum(sum_energy))

