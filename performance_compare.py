
import pickle
import numpy as np
import matplotlib.pyplot as plt

nw_size = [7] #[5, 7]

for c in nw_size:
    with open('dqn_reproduce_var_nw' + str(c) + '_beta' + str(0.0) + '_coeff1.0.pkl', 'rb') as f:


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