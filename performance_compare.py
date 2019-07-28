
import pickle
import numpy as np
import matplotlib.pyplot as plt

nw_size = [7] #[5, 7]

for c in nw_size:
    # with open('results_comparison_new/dqn_reproduce_var_nw' + str(c) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
    # with open('results_comparison_new/qtable_longer_var_nw' + str(c) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
    with open('results_comparison_new/dqn_init1_var_nw' + str(c) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
    # with open('results_comparison_new/qtable_var_nw' + str(c) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
    # with open('../Adaptive-Network-exps/jhlee_pc/20180207_beta_exp_100ep/var_nw7_beta0.0_coeff0.8.pkl', 'rb') as f:
    # with open('../Adaptive-Network-exps/minhae_desktop/one_dim_5/var_nw5_beta0.0_dim_space[5]180207-082245.pkl', 'rb') as f:
    # with open('../Adaptive-Network-exps/jhlee_pc/random/var_randomrandom_loss_nw5_beta0.0_coeff0.8.pkl', 'rb') as f:

        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        # save_reward, save_goodput, save_connect_ratio, save_energy,  save_txr, dd, aa, ss, rr = pickle.load(f)
    #     print(aa[0] * 910, ss[0], rr)
        total_reward  = save_reward
        total_goodput = save_goodput
        total_connect_ratio = save_connect_ratio
        total_energy = save_energy
        total_txr = save_txr

        print(c, len(total_connect_ratio))
        print('connectivity ratio')
        connect_ratio = []
        for i in range(len(total_connect_ratio)):
            connect_ratio += total_connect_ratio[i][150:151]
        print(np.mean(connect_ratio))
        print(np.std(connect_ratio))
        # print(sum / len(sum))

        goodput = []
        for i in range(len(total_goodput)):
            goodput += total_goodput[i][150:151]
        goodput = np.array(goodput) * 910
        print('goodput')
        print(np.mean(goodput) )
        print(np.std(goodput))

        energy = []
        for i in range(len(total_energy)):
            energy += total_energy[i][150:151]
        print('energy')
        print(np.mean(energy))
        print(np.std(energy))
        print('\n')

