
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

coeff = [0.2, 0.4, 0.6, 0.8, 1.0]
beta = [0.0, 0.1, 0.2, 0.3] #[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
patches = []
for ii in range(len(beta)):
    patch = mpatches.Patch(color=color[ii], label='beta='+ str(beta[ii]))
    patches.append(patch)

for c in range(len(beta)):
    with open('./beta_exp/var_nw7_beta' + str(beta[c]) + '_coeff0.8.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        print(beta[c], len(save_connect_ratio))
        sum = 0.
        for i in range(len(save_connect_ratio)):
            sum += np.mean(save_connect_ratio[i][140:150])
        print(sum / len(save_connect_ratio))

        sum = 0.
        for i in range(len(save_goodput)):
            sum += np.mean(save_goodput[i][140:150])
        print(sum / len(save_goodput))

        sum = 0.
        for i in range(len(save_energy)):
            sum += np.mean(save_energy[i][140:150])
        print(sum / len(save_energy))
        print('\n')

        goodput_t = np.mean(save_goodput, axis=0)
        plt.figure(0)
        plt.plot(range(len(goodput_t)), goodput_t,'-', color=color[c])
        plt.xlabel('steps')
        plt.ylabel('goodput')
        plt.legend(handles=patches)
        # plt.show()

        cr_t = np.mean(save_connect_ratio, axis=0)
        plt.figure(1)
        plt.plot(range(len(cr_t)), cr_t, '-', color=color[c])
        plt.xlabel('steps')
        plt.ylabel('connectivity ratio')
        plt.legend(handles=patches)
        # plt.show()

        energy_t = np.mean(save_energy, axis=0)
        plt.figure(2)
        plt.plot(range(len(energy_t)), energy_t, '-', color=color[c])
        plt.xlabel('steps')
        plt.ylabel('energy')
        plt.legend(handles=patches)

plt.show()
    # sum  = 0.
    # for r in save_reward:
    #     sum += np.mean(r)
    # # print(sum / 100)
    # print(np.mean(save_connect_ratio))
    # sum = 0.
    # for r in save_energy:
    #     sum += np.mean(r)
    # print(sum / 100)

