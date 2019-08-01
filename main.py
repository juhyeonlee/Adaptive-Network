
import numpy as np
import yaml
import os
import time
import matplotlib
import pickle
from matplotlib import pyplot as plt
from env import AdhocNetEnv
from Qlearning import QLearningAgent

if __name__ == '__main__':

    # set hyperparameters

    config_file = open('config/1.yaml', 'r')
    args = yaml.load(config_file)
    config_file.close()
    print(args)

    # set random seed
    np.random.seed(args['manual_seed'])

    # set action space
    n_actions = int(round((args['txr_max'] - args['txr_min']) / args['txr_step'])) + 1
    action_space = [round(args['txr_min'] + i * args['txr_step'], 1) for i in range(n_actions)]
    action_space = np.array(action_space)
    
    # se
    ep_length = args['ep_length']
    num_ep = args['num_ep']

    learning_rate = args['lr']
    discount_factor = args['gamma']
    beta = args['beta']

    save_goodput = []
    save_reward = []
    save_energy = []
    save_connect_ratio = []
    save_txr = []
    # save_num_players = []
    #
    # if not os.path.exists('./model'):
    #     os.makedirs('./model')

    env = AdhocNetEnv(action_space, args)

    t = time.time()

    for ep in range(num_ep):
        print("num ep", ep)
        state = env.reset()
        num_players = env.num_players
        agent = []
        for i in range(len(state) - 4):
            agent.append(QLearningAgent(action_space, args))
        action = np.zeros(len(state), dtype=np.float32)
        # q_value = np.zeros((len(state), n_actions), dtype=np.float32)
        goodput_trace = []
        reward_trace = []
        energy_trace = []
        con_ratio_trace = []
        txr_trace = []
        qvalue_trace = []
        sum_reward = 0.0
        for steps in range(ep_length):
            print('step number: ', steps)
            for i in range(len(state) - 4):
                action[i] = agent[i].get_action(state[i], steps)
            # print('action check:', action)
            next_state, reward, goodput, energy, con_ratio = env.step(action, steps, ep)


            for i in range(len(state) - 4):
                agent[i].learn(state[i], action[i], reward[i], next_state[i])

            state = next_state
            sum_reward += np.mean(reward)
            goodput_trace.append(goodput)
            reward_trace.append(sum_reward)
            energy_trace.append(np.sum(energy)/num_players) #energy per an agent
            #print('energy trace: ', energy_trace)
            con_ratio_trace.append(con_ratio)
            # txr_trace.append(env.actiontxr)
            print(goodput, sum_reward, con_ratio, np.sum(energy)/num_players)


        # test
        for i in range(len(state) - 4):
            action[i] = agent[i].get_greedy_action(state[i])
        next_state, reward, goodput, energy, con_ratio = env.step(action, ep_length, ep)
        goodput_trace.append(goodput)
        reward_trace.append(np.mean(reward))
        energy_trace.append(np.sum(energy)/num_players) #energy per an agent
        con_ratio_trace.append(con_ratio)
        print('test performance', goodput, con_ratio,  np.sum(energy)/num_players)
        # txr_trace.append(env.txr)

        save_goodput.append(goodput_trace)
        save_reward.append(reward_trace)
        save_energy.append(energy_trace)
        save_connect_ratio.append(con_ratio_trace)
        save_txr.append(txr_trace)
        # save_num_players.append(num_players)
        # saver.save(sess, './model/model_ep'+ str(episode) + '_nw' + str(one_dim) + '_beta' + str(beta) + '_coeff' + str(utility_coeff) + '.ckpt')

        # goodput_trace.append(goodput)
        # reward_trace.append(np.mean(reward))
        # energy_trace.append(np.sum(energy))
        # print('greedy approach - TX range: ', env.txr)

        # print('goodput trace: ', goodput_trace)
        # print('reward trace :', reward_trace)

        # plt.clf()
        plt.figure(0)
        plt.plot(range(ep_length+1), goodput_trace,'-')
        plt.xlabel('Steps')
        plt.ylabel('Goodput')
        # plt.savefig('fig1_one_trial_goodput' +str(ep)+ '.eps')


        # plt.clf()
        plt.figure(11)
        plt.plot(range(ep_length+1), energy_trace,'-')
        plt.xlabel('Steps')
        plt.ylabel('Energy')
        # plt.savefig('fig1_one_trial_energy.eps')


        # plt.clf()
        plt.figure(2)
        plt.plot(range(ep_length+1), con_ratio_trace,'-')
        plt.xlabel('Steps')
        plt.ylabel('Connectivity Ratio')
        # plt.savefig('fig1_one_trial_cr' +str(ep)+ '.eps')


        #
        #
        # plt.clf()
        plt.figure(3)
        plt.plot(range(ep_length+1), reward_trace,'-')
        plt.xlabel('Steps')
        plt.ylabel('Cumulative Reward')
        # plt.savefig('fig1_one_trial_reward' +str(ep)+ '.eps')

        # plt.clf()
        # plt.figure(4)
        # plt.plot(range(ep_length), qvalue_trace,'-')
        # plt.xlabel('Stpes')
        # plt.ylabel('Q value')
        # plt.savefig('fig1_one_trial_qvalue' +str(ep)+ '.eps')

        print('processing time ', time.time() - t)
        # plt.show()


        # txr_mat = save_txr[0]  # np.mean(save_txr, axis=0)
        # txr_colormat = []
        # for i in range(len(txr_mat)):
        #     txr_colormat.append(txr_mat[i][:-4])
        # plt.clf()
        # plt.figure(5)
        # plt.matshow(txr_colormat, cmap='pink', aspect=0.5)  # defaults
        # plt.xlabel('The Index of Nodes')
        # plt.ylabel('Steps')
        # a = plt.colorbar(fraction=0.046, pad=0.04)
        # a.set_label('Radius of Transmission Range')
        # plt.savefig('fig1_one_trial_color' +str(episode)+ '.eps')
        # plt.show()
    with open('qtable_gamma0.1_short_more_var_nw' + str(args['nw_size']) + '_beta' + str(beta) + '_coeff' + str(args['utility_coeff']) + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([save_reward, save_goodput, save_connect_ratio, save_energy, save_txr], f)

    # print('average goodput: ', np.sum(sum_goodput), 'average reward :', np.sum(sum_energy))

