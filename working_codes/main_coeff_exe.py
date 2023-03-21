
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle
import argparse
import time


from env import Environment
from DQNAgent import DQNAgent


if __name__ == '__main__':

    # experiment
    # one_dim_set = [6, 8, 10, 20, 30]
    # utility_coeff_set = [1.0, 3.0, 5.0, 7.0]
    # beta_set = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--nw_size", help="network size", type=int, default=5)
    parser.add_argument("--coeff", help="utility coeff", type=float, default=0.8)
    parser.add_argument("--beta", help="beta, link failure rate", type=float, default=0.0)
    args = parser.parse_args()

    save_goodput = []
    save_reward = []
    save_energy = []
    save_connect_ratio = []
    save_txr = []
    save_num_players = []
    mean_goodput =[]
    mean_energy = []
    mean_con_ratio = []

    t_start = time.time()

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
    ep_length = 110#00
    num_ep = 500#000

    learning_rate = 0.01
    discount_factor = 0.7

    beta = args.beta
    utility_coeff_space = [0.0, 0.2,  0.4, 0.6, 0.8, 1.0]
    ########################

    mean_goodput = []# np.zeros(len(utility_coeff_space), dtype=np.float32)
    mean_energy = []#np.zeros(len(utility_coeff_space), dtype=np.float32)
    mean_con_ratio = []#np.zeros(len(utility_coeff_space), dtype=np.float32)

    for coef_idx, utility_coeff in enumerate(utility_coeff_space) :

        mean_goodput_trace = []
        mean_con_ratio_trace = []
        mean_energy_trace = []

        env = Environment(one_dim, mu, init_txr, utility_coeff, utility_pos_coeff, action_space)

        n_actions = env.n_actions
        action_space = env.action_space


        for episode in range(num_ep):




            state, num_players = env.reset(init_txr, beta) # len(state) =  num_players + 4
            agent = []
            tf.reset_default_graph()
            sess = tf.Session()

            t_episode_start = time.time()

            if episode > 0:
                print('reset time duration', t_episode_end - t_episode_start)

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
                print('utility_coeff: ', utility_coeff, 'episode: ',episode, 'step number: ', steps)
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

                #collection for all steps , singel episode
                goodput_trace.append(goodput)
                reward_trace.append(np.mean(reward))
                energy_trace.append(np.sum(energy)/num_players) #energy per an agent
                #print('energy trace: ', energy_trace)
                con_ratio_trace.append(con_ratio)
                txr_trace.append(env.txr)
                qvalue_trace.append(np.mean(q_value))

            # mean value for one episode, collection for all episode (single coeff)
            mean_goodput_trace.append(np.mean(goodput_trace[epsilon['epsilon_step']:ep_length]))
            mean_energy_trace.append(np.mean(energy_trace[epsilon['epsilon_step']:ep_length]))
            mean_con_ratio_trace.append(np.mean(con_ratio_trace[epsilon['epsilon_step']:ep_length]))

            # # test
            # for i in range(len(state) - 4):
            #     action[i] = agent[i].get_greedy_action(state[i])
            # next_state, reward, goodput, energy, con_ratio = env.step(action, ep_length)
            # goodput_trace.append(goodput)
            # reward_trace.append(np.mean(reward))
            # energy_trace.append(np.sum(energy)/num_players) #energy per an agent
            # con_ratio_trace.append(con_ratio)
            # txr_trace.append(env.txr)
            # collection for all episode , single coeff
            save_goodput.append(goodput_trace)
            save_reward.append(reward_trace)
            save_energy.append(energy_trace)
            save_connect_ratio.append(con_ratio_trace)
            save_txr.append(txr_trace)
            save_num_players.append(num_players)

            # tensor flow parameter saver
            saver.save(sess, './model/model_ep'+ str(episode) + '_nw' + str(one_dim) + '_beta' + str(beta) + '_coeff' + str(utility_coeff) + '_ep' + str(num_ep) + '_step' +str(ep_length-epsilon['epsilon_step'])+str(time.strftime("%y%m%d-%H%M%S")) + '.ckpt')
            t_episode_end = time.time()

        # for every ceoff
        mean_goodput.append(np.mean(mean_goodput_trace))
        mean_con_ratio.append(np.mean(mean_con_ratio_trace))
        mean_energy.append(np.mean(mean_energy_trace))

            # goodput_trace.append(goodput)
            # reward_trace.append(np.mean(reward))
            # energy_trace.append(np.sum(energy))
            # print('greedy approach - TX range: ', env.txr)

            # print('goodput trace: ', goodput_trace)
            # print('reward trace :', reward_trace)

            # plt.figure(0)
            # plt.plot(range(ep_length+1), goodput_trace,'-*')
            # plt.xlabel('episode')
            # plt.ylabel('goodput')
            # plt.title('coef:', utility_coeff)
            # #plt.show()
            #
            # plt.figure(1)
            # plt.plot(range(ep_length+1), energy_trace,'-+')
            # plt.xlabel('episode')
            # plt.ylabel('energy per an agent')
            # plt.title('coef:', utility_coeff)
            #
            #
            # plt.figure(2)
            # plt.plot(range(ep_length+1), con_ratio_trace,'-+')
            # plt.xlabel('episode')
            # plt.ylabel('connectivity ratio')
            # plt.title('coef:', utility_coeff)

            #plt.show()
            #
            #
            # plt.figure(3)
            # plt.plot(range(ep_length+1), reward_trace,'-+')
            # plt.xlabel('episode')
            # plt.ylabel('reward')
            # plt.title('coef:', utility_coeff)
            #
            # plt.figure(4)
            # plt.plot(range(ep_length), qvalue_trace,'-+')
            # plt.xlabel('episode')
            # plt.ylabel('Q value')
            # plt.title('coef:', utility_coeff)
            # plt.show()

    print('goodput: ', mean_goodput, 'energy', mean_energy, 'connectivity', mean_con_ratio)
    print('processing time ', time.time() - t_start)


    # Saving the objects:
    with open('var_nw' + str(one_dim) + '_beta' + str(beta) + '_coeff_space' + str(utility_coeff_space) + str(time.strftime("%y%m%d-%H%M%S")) +  '.pkl', 'wb') as f:  # Python 3: open(..., 'wb') # wb: write binary #rb: read binary
        pickle.dump([save_num_players, save_reward, save_goodput, save_connect_ratio, save_energy, save_txr, mean_goodput, mean_energy, mean_con_ratio], f)

    # print('average goodput: ', np.sum(sum_goodput), 'average reward :', np.sum(sum_energy))


    plt.figure(0)
    plt.plot(utility_coeff_space, mean_goodput, '-*')
    plt.xlabel('coeff')
    plt.ylabel('goodput')
    #plt.title('coef:', utility_coeff)
    # plt.show()
    plt.figure(1)
    plt.plot(utility_coeff_space, mean_energy, '-*')
    plt.xlabel('coeff')
    plt.ylabel('energy')
    #plt.title('coef:', utility_coeff)
    # plt.show()
    plt.figure(2)
    plt.plot(utility_coeff_space, mean_con_ratio, '-*')
    plt.xlabel('coeff')
    plt.ylabel('connectivity')
    #plt.title('coef:', utility_coeff)
    plt.show()

