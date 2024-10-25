# synth_prob_samples.py
# Written Ian Rankin - October 2024
#
# A set of code to show an example usage of the synthetic user class.
# Additionally, it allows testing the class with different examples


import numpy as np
import random
import pickle
import argparse
import lop
import matplotlib.pyplot as plt
from tqdm import tqdm

import pdb

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../../experiments')

from experiment_helper import get_synth_user, get_fake_func



def main():
    parser = argparse.ArgumentParser(description='bimodal example with different models and active learners')
    parser.add_argument('--pickle', type=str, default='../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    #parser.add_argument('--env', type=int, default=0, help='Environment number')
    parser.add_argument('--fake_func', type=str, default='min', help='Selects fake function type [linear, squared, logistic, sin_exp, min, max, squared_min_max]')
    parser.add_argument('--user', type=str, default='human_choice', help='Selects the type of user [human_choice perfect]')
    parser.add_argument('--p_pair', type=float, default=0.95, help='value to tune synth user to for choose comparisions')
    #parser.add_argument('--p_abs', type=float, default=0.95, help='value to tune synth user to for ratings')
    parser.add_argument('--num_alts', type=int, default=2, help='Number of alternative assumed to be passed to user')
    args = parser.parse_args()

    dim_rewards = 5

    ###### Open and read path data from Pickle
    with open(args.pickle, 'rb') as f:
        path_data = pickle.load(f)
    
    num_train = -1
    num_eval = -1
    train_data = [path_d[:num_train] for path_d in path_data['train']]
    eval_data = [path_d[:num_eval] for path_d in path_data['eval']]


    # Finish loading data!

    envs = [0,1,2,3,4,5,6,7,8,9]
    p_abss = [0.7, 0.95]
    N_reps = 30
    #envs = [3,2]

    all_acc_c = np.empty((len(p_abss), len(envs), N_reps))
    all_acc_r = np.empty((len(p_abss), len(envs), N_reps))
    all_acc_c_pro = np.empty((len(p_abss), len(envs), N_reps))
    all_acc_r_pro = np.empty((len(p_abss), len(envs), N_reps))

    for i_abs, p_abs in enumerate(p_abss):
        print('P_ABS=' +str(p_abs))
        for i_env, env in enumerate(envs):
            

            for i_reps in range(N_reps):

                # Read in data
                eval_env_d = eval_data[env]
                train_env_d = train_data[env]

                eval_user_d = np.empty((0, dim_rewards))
                for path_d in eval_env_d:
                    eval_user_d = np.append(eval_user_d, path_d['rewards'], axis=0)

                train_user_d = np.empty((0, dim_rewards))
                for path_d in train_env_d:
                    train_user_d = np.append(train_user_d, path_d['rewards'], axis=0)



                ###### Create fake function and synthetic user
                config = {'dim_rewards': dim_rewards}

                fake_f = get_fake_func(args.fake_func, config)
                user_f = get_synth_user(args.user, fake_f, config)

                print('Starting learning betas for')
                while True:
                    try:
                        print('Running tune beta')
                        user_f.learn_beta(eval_user_d, args.p_pair, Q_size=2, p_sigma=p_abs)
                        print('Completed, breaking out of loop')
                        break
                    except:
                        print('Unable to tune beta for user synth')
                        # gonna randomize the function and try again
                        fake_f.randomize()

                print('Done, ready to test method')


                ### Test user synth

                N = 2500

                count_choose = 0
                count_rate = 0

                for i in range(N):
                    pair = np.random.choice(train_user_d.shape[0], 2, replace=False)

                    idx = user_f.choose(train_user_d[pair])
                    idx_rate = np.argmax(user_f.rate(train_user_d[pair]))
                    corr_idx = np.argmax(user_f.fake_f(train_user_d[pair]))

                    if corr_idx == idx:
                        count_choose += 1

                    if idx_rate == idx:
                        count_rate += 1 

                acc_rate = count_rate / N
                acc = count_choose / N


                ##### Test given each planning problem problem instance
                pro_acc_rate = np.empty(len(train_env_d))
                pro_acc = np.empty(len(train_env_d))

                for pro_i in range(len(train_env_d)):
                    N_i = int(np.ceil(N / len(train_env_d)))

                    count_choose = 0
                    count_rate = 0
                    pro_rewards = train_env_d[pro_i]['rewards']
                    for i in range(N_i):
                        pair = np.random.choice(len(pro_rewards), 2, replace=False)

                        idx = user_f.choose(pro_rewards[pair])
                        idx_rate = np.argmax(user_f.rate(pro_rewards[pair]))
                        corr_idx = np.argmax(user_f.fake_f(pro_rewards[pair]))

                        if corr_idx == idx:
                            count_choose += 1

                        if idx_rate == idx:
                            count_rate += 1 
                    
                    pro_acc_rate[pro_i] = count_rate / N_i
                    pro_acc[pro_i] = count_choose / N_i

                    print('problem: ' + str(pro_i) + ' acc_rate = ' + str(pro_acc_rate[pro_i]) + ' acc_choose = ' + str(pro_acc[pro_i]))


                # a_r[j] = acc_rate
                # a_c[j] = acc
                all_acc_c[i_abs, i_env, i_reps] = acc
                all_acc_r[i_abs, i_env, i_reps] = acc_rate
                all_acc_c_pro[i_abs, i_env, i_reps] = np.mean(pro_acc)
                all_acc_r_pro[i_abs, i_env, i_reps] = np.mean(pro_acc_rate)


            print('\n\n\n\n\n\n Env ' + str(env))
            print('\tEnv: ' +str(env)+ ' Eval Accuracy of choose is = ' + str(acc))
            print('\tEnv: ' +str(env)+ ' Eval Accuracy of rate is = ' + str(acc_rate))
            print('\tEnv: ' +str(env)+ ' Prob Accuracy of choose is = ' + str(np.mean(pro_acc)) + ' std: ' + str(np.std(pro_acc)))
            print('\tEnv: ' +str(env)+ ' Prob Accuracy of rate is = ' + str(np.mean(pro_acc_rate)) + ' std: ' + str(np.std(pro_acc_rate)))


    np.savez('p_samples.npz', all_acc_c=all_acc_c, all_acc_r=all_acc_r, all_acc_c_pro=all_acc_c_pro, all_acc_r_pro=all_acc_r_pro)
    # Plot results

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

    plt.legend(legs)

    plt.show()


if __name__ == '__main__':
    main()
