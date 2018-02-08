import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

coeff = 0.8 #[0.2, 0.4, 0.6, 0.8, 1.0]
beta = 0.0 #[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
nw_size= 7#[7, 12, 17]


with open('../Adaptive-Network-exps/jhlee_pc/20180206_beta_exp_50ep/var_nw' + str(nw_size) + '_beta' + str(beta) + '_coeff0.8.pkl', 'rb') as f:
    save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)



    txr_mat = save_txr[0]#np.mean(save_txr, axis=0)
    txr_colormat = []
    for i in range(len(txr_mat)):
        txr_colormat.append(txr_mat[i][:-4])

    # #
    # fig, ax = plt.subplots()
    # heatmap = ax.pcolor(txr_mat, cmap=plt.cm.Blues)
    #
    # # want a more natural, table-like display
    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    #
    # ax.set_xticklabels('nodes', minor=False)
    # ax.set_yticklabels('steps', minor=False)
    # # plt.colorbar()
    # plt.show()

    plt.matshow(txr_colormat, cmap='pink')  # defaults
    plt.xlabel('nodes (agents)')
    plt.ylabel('steps')
    plt.colorbar()
    plt.show()

    # plt.matshow(txr_mat, vmin=0, vmax=99)  # same
    # plt.matshow(A, vmin=10, vmax=90)