
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

nw_size= [7, 12, 17, 22]
nw_label = ['1.0x' + r'$10^4$ ' + '[' + r'$m^2$' +']', '4.0x' + r'$10^4$ ' + '[' + r'$m^2$'+']', '9.0x' + r'$10^4$ '
            + '[' + r'$m^2$'+']', '1.6x' + r'$10^5$ ' + '[' + r'$m^2$'+']']
color = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']

nw_patches = []
for ii in range(len(nw_size)):
    patch = mpatches.Patch(color=color[ii], label='Size=' + str(nw_label[ii]))
    nw_patches.append(patch)

num_player = []
# network exp
total_reward = []
total_goodput = []
total_connect_ratio = []
total_energy = []
total_txr = []
with open('../Adaptive-Network-exps/minhae_desktop/one_dim_5/var_nw5_beta0.0_dim_space[5]180207-082245.pkl', 'rb') as kk:
    save_rewardk, save_goodputk, save_connect_ratiok, save_energyk, dd, save_txrk, dd2, dd3, dd5 = pickle.load(kk)
    num_player_per_nw5 = []
    for i in range(20):
        num_player_per_nw5.append(len(save_txrk[i][0]))
    num_player.append(num_player_per_nw5)

goodput_data = pd.DataFrame(columns=['timestep', 'goodput', 'hue'])
connect_data = pd.DataFrame(columns=['timestep', 'connect', 'hue'])
enery_data = pd.DataFrame(columns=['timestep', 'energy', 'hue'])

for c in range(len(nw_size)):
    with open('../Adaptive-Network-exps/amazon/20180206_nwsize_exp_10ep/var_nw' + str(nw_size[c]) + '_beta' + str(0.0) + '_coeff0.8.pkl', 'rb') as f:
        save_reward, save_goodput, save_connect_ratio, save_energy, save_txr = pickle.load(f)
        total_reward = save_reward
        total_goodput = save_goodput
        total_connect_ratio = save_connect_ratio
        total_energy = save_energy
        total_txr = save_txr


    with open('../Adaptive-Network-exps/amazon/20180209_nwsize_exp_10ep/var_nw' + str(nw_size[c]) + '_beta0.0_coeff0.8.pkl', 'rb') as ff:

        save_reward2, save_goodput2, save_connect_ratio2, save_energy2, save_txr2 = pickle.load(ff)
        total_reward += save_reward2
        total_goodput += save_goodput2
        total_connect_ratio += save_connect_ratio2
        total_energy += save_energy2
        total_txr += save_txr2

        print(len(total_goodput))
        num_player_per_nw = []
        for i in range(len(total_txr)):
            num_player_per_nw.append(len(total_txr[i][0]))

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


    total_goodput = np.array(total_goodput) * 910
    total_connect_ratio = np.array(total_connect_ratio)
    total_energy = np.array(total_energy)
    # goodput_t = np.mean(total_goodput, axis=0)
    # total_goodput = np.transpose(total_goodput, [1, 0])
    # goodput_t = goodput_t * 910
    for i in range(total_goodput.shape[0]):
        for tt in range(total_goodput.shape[1]):
            goodput_data = goodput_data.append({"timestep": tt, "goodput": total_goodput[i, tt], 'hue': color[c]}, ignore_index=True)
            connect_data = connect_data.append({'timestep': tt, "connect": total_connect_ratio[i, tt], 'hue': color[c]}, ignore_index=True)
            enery_data = enery_data.append({'timestep': tt, "energy": total_energy[i, tt], 'hue': color[c]},
                                               ignore_index=True)



    energy_t = np.mean(total_energy, axis=0)


    num_player.append(num_player_per_nw)

plt.figure(0)
sns.lineplot(x="timestep", y="goodput", hue='hue', data=goodput_data)
# plt.errorbar(range(len(goodput_t)), goodput_t, xerr=0.5, yerr=2*np.std(total_goodput_origin, axis=0))
plt.xlabel('Steps')
plt.ylabel('Goodput [Mbps]')
plt.legend(handles=nw_patches)#, loc='lower right')
#plt.ylim((0, 350))
# plt.legend(handles=beta_patches)
# plt.show()
plt.savefig('fig4_nw_goodput.pdf')

plt.figure(1)
# plt.plot(range(len(cr_t)), cr_t, '-', color=color[c])
sns.lineplot(x="timestep", y="connect", hue='hue', data=connect_data)
plt.xlabel('Steps')
plt.ylabel('Connectivity Ratio')
# plt.legend(handles=beta_patches)
plt.legend(handles=nw_patches)
# plt.show()
plt.savefig('fig4_nw_cr.pdf')

plt.figure(2)
sns.lineplot(x="timestep", y="energy", hue='hue', data=enery_data)
plt.xlabel('Steps')
plt.ylabel('Energy Per Node [dBm]')
# plt.legend(handles=beta_patches)
plt.legend(handles=nw_patches)
plt.savefig('fig4_nw_energy.pdf')

plt.figure(3)
aa = plt.boxplot(num_player, 0, 'k+')
plt.setp(aa['medians'], color='k')
plt.xticks([1, 2, 3, 4, 5], ['6.4 x ' + r'$10^3$', '1.0 x ' + r'$10^4$', '4.0 x ' + r'$10^4$', '9.0 x ' + r'$10^4$', '1.6 x ' + r'$10^5$']) #[100, 400, 900, 1600]
plt.xlabel('Network Size (Area) ['+ r'$m^2$' + ']')
plt.ylabel('The Number of Nodes')
plt.savefig('fig4_nw_box.pdf')


plt.show()