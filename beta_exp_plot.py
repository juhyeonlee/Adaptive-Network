
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

coeff = [0.2, 0.4, 0.6, 0.8, 1.0]
beta = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
nw_size= [7] #, 12, 17, 22]
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
beta_patches = []
for ii in range(len(beta)):
    patch = mpatches.Patch(color=color[ii], label= r'$\beta$' + '='+ str(beta[ii]))
    beta_patches.append(patch)

nw_patches = []
for ii in range(len(nw_size)):
    patch = mpatches.Patch(color=color[ii], label='size='+ str(nw_size[ii]))
    nw_patches.append(patch)

# patch = mpatches.Patch(color=color[0], label='DQN')
# nw_patches.append(patch)
# patch = mpatches.Patch(color=color[8], label='Random')
# nw_patches.append(patch)

# # network exp
# for c in range(len(nw_size)):
#     with open('../Adaptive-Network-exps/amazon/20180206_nwsize_exp_10ep/var_nw' + str(nw_size[c]) + '_beta' + str(beta[0]) + '_coeff0.8.pkl', 'rb') as f:
#         save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
#         print(nw_size[c], len(save_connect_ratio))
#         sum = 0.
#         for i in range(len(save_connect_ratio)):
#             sum += np.mean(save_connect_ratio[i][100:150])
#         print(sum / len(save_connect_ratio))
#
#         sum = 0.
#         for i in range(len(save_goodput)):
#             sum += np.mean(save_goodput[i][100:150])
#         print(sum / len(save_goodput))
#
#         sum = 0.
#         for i in range(len(save_energy)):
#             sum += np.mean(save_energy[i][100:150])
#         print(sum / len(save_energy))
#         print('\n')
#
#         goodput_t = np.mean(save_goodput, axis=0)
#         plt.figure(0)
#         plt.plot(range(len(goodput_t)), goodput_t,'-', color=color[c])
#         plt.xlabel('steps')
#         plt.ylabel('goodput')
#         # plt.legend(handles=beta_patches)
#         plt.legend(handles=nw_patches)
#         # plt.show()
#
#         cr_t = np.mean(save_connect_ratio, axis=0)
#         plt.figure(1)
#         plt.plot(range(len(cr_t)), cr_t, '-', color=color[c])
#         plt.xlabel('steps')
#         plt.ylabel('connectivity ratio')
#         # plt.legend(handles=beta_patches)
#         plt.legend(handles=nw_patches)
#         # plt.show()
#
#         energy_t = np.mean(save_energy, axis=0)
#         plt.figure(2)
#         plt.plot(range(len(energy_t)), energy_t, '-', color=color[c])
#         plt.xlabel('steps')
#         plt.ylabel('energy')
#         # plt.legend(handles=beta_patches)
#         plt.legend(handles=nw_patches)
# beta exp
total_reward = []
total_goodput = []
total_connect_ratio = []
total_energy = []
total_txr = []
for c in range(len(beta)):
    with open('../Adaptive-Network-exps/jhlee_pc/20180206_beta_exp_50ep/var_nw' + str(7) + '_beta' + str(beta[c]) + '_coeff0.8.pkl', 'rb') as f:
        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        total_reward  = save_reward
        total_goodput = save_goodput
        total_connect_ratio = save_connect_ratio
        total_energy = save_energy
        total_txr = save_txr

    with open('../Adaptive-Network-exps/jhlee_pc/20180207_beta_exp_100ep/var_nw' + str(7) + '_beta' + str(
            beta[c]) + '_coeff0.8.pkl', 'rb') as ff:

        save_reward2, save_goodput2, save_connect_ratio2, save_energy2, save_txr2 = pickle.load(ff)
        total_reward += save_reward2
        total_goodput += save_goodput2
        total_connect_ratio += save_connect_ratio2
        total_energy += save_energy2
        total_txr += save_txr2

        # with open('var_nw7_beta0.0_coeff0.8_10ep_fornw.pkl',
        #           'wb') as fff:  # Python 3: open(..., 'wb')
        #     pickle.dump([save_reward[0:10], save_goodput[0:10], save_connect_ratio[0:10], save_energy[0:10], save_txr[0:10]], fff)

        print(beta[c], len(total_connect_ratio))
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

    goodput_t = np.mean(total_goodput, axis=0)
    plt.figure(0)
    plt.plot(range(len(goodput_t)), goodput_t * 910,'-', color=color[c])
    plt.xlabel('Steps')
    plt.ylabel('Goodput [Mbps]')
    # plt.legend(handles=beta_patches)
    plt.legend(handles=beta_patches)
    # plt.show()
    plt.savefig('fig3_beta_goodput.eps')

    cr_t = np.mean(total_connect_ratio, axis=0)
    plt.figure(1)
    plt.plot(range(len(cr_t)), cr_t, '-', color=color[c])
    plt.xlabel('Steps')
    plt.ylabel('Connectivity Ratio')
    # plt.legend(handles=beta_patches)
    plt.legend(handles=beta_patches)
    # plt.show()
    plt.savefig('fig3_beta_cr.eps')

    energy_t = np.mean(total_energy, axis=0)
    plt.figure(2)
    plt.plot(range(len(energy_t)), energy_t, '-', color=color[c])
    plt.xlabel('Steps')
    plt.ylabel('Energy Per Node [dBm]')
    # plt.legend(handles=beta_patches)
    plt.legend(handles=beta_patches)
    plt.savefig('fig3_beta_energy.eps')

# # random
# with open('../Adaptive-Network-exps/jhlee_pc/random/var_randomrandom_nw7_beta0.0_coeff0.8.pkl', 'rb') as f:
#     ave_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
#     sum = 0.
#     print(len(save_goodput))
#     for i in range(len(save_connect_ratio)):
#         sum += np.mean(save_connect_ratio[i][100:150])
#     print(sum / len(save_connect_ratio))
#
#     sum = 0.
#     for i in range(len(save_goodput)):
#         sum += np.mean(save_goodput[i][100:150])
#     print(sum / len(save_goodput))
#
#     sum = 0.
#     for i in range(len(save_energy)):
#         sum += np.mean(save_energy[i][100:150])
#     print(sum / len(save_energy))
#     print('\n')
#     goodput_t = np.mean(save_goodput, axis=0)
#     plt.figure(0)
#     plt.plot(range(len(goodput_t)), goodput_t, '-', color=color[8])
#     plt.xlabel('steps')
#     plt.ylabel('goodput')
#     # plt.legend(handles=beta_patches)
#     plt.legend(handles=nw_patches)
#     # plt.show()
#
#     cr_t = np.mean(save_connect_ratio, axis=0)
#     plt.figure(1)
#     plt.plot(range(len(cr_t)), cr_t, '-', color=color[8])
#     plt.xlabel('steps')
#     plt.ylabel('connectivity ratio')
#     # plt.legend(handles=beta_patches)
#     plt.legend(handles=nw_patches)
#     # plt.show()
#
#     energy_t = np.mean(save_energy, axis=0)
#     plt.figure(2)
#     plt.plot(range(len(energy_t)), energy_t, '-', color=color[8])
#     plt.xlabel('steps')
#     plt.ylabel('energy')
#     # plt.legend(handles=beta_patches)
#     plt.legend(handles=nw_patches)


# plt.show()
    # sum  = 0.
    # for r in save_reward:
    #     sum += np.mean(r)
    # # print(sum / 100)
    # print(np.mean(save_connect_ratio))
    # sum = 0.
    # for r in save_energy:
    #     sum += np.mean(r)
    # print(sum / 100)

