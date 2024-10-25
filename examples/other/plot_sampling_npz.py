import numpy as np
import matplotlib.pyplot as plt


data = np.load('p_samples.npz')
# Plot results

envs = [0,1,2,3,4,5,6,7,8,9]
p_abss = [0.7, 0.95]

all_acc_c = data['all_acc_c']
all_acc_r = data['all_acc_r']
all_acc_c_pro = data['all_acc_c_pro']
all_acc_r_pro = data['all_acc_r_pro']


ax = plt.gca()

colors = ['red', 'blue', 'cyan', 'orange']
styles = ['-', '-.']
legs = []

vals = [all_acc_c, all_acc_r, all_acc_c_pro,all_acc_r_pro]
sig_plot = 1

xs = np.arange(len(envs))
for i in range(len(p_abss)):
    for j in range(len(vals)):
        mean_ij = np.mean(vals[j][i], axis=1)
        std_ij = np.std(vals[j][i], axis=1)
        ax.fill_between(
            xs,
            mean_ij-std_ij*sig_plot,
            mean_ij+std_ij*sig_plot,
            color=colors[j],
            alpha=0.1,
            label='_nolegend_'
        )

        ax.plot(xs, mean_ij, color=colors[j], linestyle=styles[i])

    s = 'p_abs='+str(p_abss[i]) + ' '
    legs += [s+'acc_choice', s+'acc_rate', s+'pro_acc_choice', s+'pro_acc_rate']

plt.plot([0,9], [0.95, 0.95], color='black', linestyle='--')
plt.plot([0,9], [0.7, 0.7], color='black', linestyle='--')
legs += ['0.95 prob', '0.7 prob']

plt.legend(legs)



plt.show()