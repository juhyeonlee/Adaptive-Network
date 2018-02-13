import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import figaspect, Figure

coeff = 0.8 #[0.2, 0.4, 0.6, 0.8, 1.0]
beta = 0.0 #[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
nw_size= 7#[7, 12, 17]


with open('./2onetime_var_nw7_beta0.0_coeff0.8.pkl', 'rb') as f:
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
    # plt.figure(0)
    plt.matshow(txr_colormat, cmap='pink', aspect=0.4)  # defaults
    plt.xlabel('The Index of Nodes')
    plt.ylabel('Steps')
    a = plt.colorbar(fraction=0.04, pad=0.04)
    a.set_label('Radius of Transmission Range')
    plt.savefig('fig1_one_trial_color_good' + '.eps', bbox_inches='tight')
    plt.show()
    # plt.matshow(txr_mat, vmin=0, vmax=99)  # same
    # plt.matshow(A, vmin=10, vmax=90)