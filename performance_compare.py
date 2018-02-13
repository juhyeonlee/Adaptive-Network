
import pickle
import numpy as np
import matplotlib.pyplot as plt

nw_size = [5, 7]

for c in range(len(nw_size)):
    #with open('../Adaptive-Network-exps/amazon/20180207_qtable_100ep/qt_var_nw' + str(nw_size[c]) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
    with open('./var_randomrandom_loss_nw5_beta0.0_coeff0.8.pkl', 'rb') as f:
        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        total_reward  = save_reward
        total_goodput = save_goodput
        total_connect_ratio = save_connect_ratio
        total_energy = save_energy
        total_txr = save_txr

        print(nw_size[c], len(total_connect_ratio))
        sum = 0.
        for i in range(len(total_connect_ratio)):
            sum += np.mean(total_connect_ratio[i][100:150])
        print(sum / len(total_connect_ratio))

        sum = 0.
        for i in range(len(total_goodput)):
            sum += np.mean(total_goodput[i][100:150])
        print(sum / len(total_goodput))

        sum = 0.
        for i in range(len(total_energy)):
            sum += np.mean(total_energy[i][100:150])
        print(sum / len(total_energy))
        print('\n')