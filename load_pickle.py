
import pickle
import numpy as np

with open('var_nw7_beta0.0_coeff20.ckpt.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
    print(save_goodput)
    sum = 0.
    for i in range(len(save_connect_ratio)):
        sum += save_connect_ratio[i][500]
        print(len(save_connect_ratio))
    print(sum / len(save_connect_ratio))
    # sum  = 0.
    # for r in save_reward:
    #     sum += np.mean(r)
    # # print(sum / 100)
    # print(np.mean(save_connect_ratio))
    # sum = 0.
    # for r in save_energy:
    #     sum += np.mean(r)
    # print(sum / 100)