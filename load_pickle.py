
import pickle
import numpy as np
coeff = [0.0, 0.2, 0.4]
for i in range(len(coeff)):
    with open('../Adaptive-Network-exps/jhlee_pc/20180205/var_nw7_beta0.0_coeff' +str(coeff[i]) + '.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        print("coeff: ", coeff[i])
        sum = 0.
        for i in range(len(save_connect_ratio)):
            sum += np.mean(save_connect_ratio[i][100:200])
        print("connect_ratio: ", sum / len(save_connect_ratio))

        sum = 0.
        for i in range(len(save_goodput)):
            sum += np.mean(save_goodput[i][100:200])
        print("goodput: ", sum / len(save_goodput))

        sum = 0.
        for i in range(len(save_energy)):
            sum += np.mean(save_energy[i][100:200])
        print("energy: ", sum / len(save_energy))
        print('\n')
        # sum  = 0.
        # for r in save_reward:
        #     sum += np.mean(r)
        # # print(sum / 100)
        # print(np.mean(save_connect_ratio))
        # sum = 0.
        # for r in save_energy:
        #     sum += np.mean(r)
        # print(sum / 100)