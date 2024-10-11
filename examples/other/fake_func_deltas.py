# synth_user_example.py
# Written Ian Rankin - September 2024
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


from sklearn_extra.cluster import KMedoids
def downsample_kmed_rand(X, num_downselect, kmed_pct=0.5):
    num_kmed = int(kmed_pct * num_downselect)
    num_rand = num_downselect - num_kmed
    
    rand_idxs = np.random.choice(X.shape[0], num_rand, replace=False)
    non_rand_idxs = list(set(list(range(X.shape[0]))) - set(rand_idxs))
    
    cent_pts = KMedoids(n_clusters=num_kmed).fit(X[non_rand_idxs]).cluster_centers_

    return np.append(cent_pts, X[rand_idxs], axis=0)


        

def test_trained_model2(fake_f, down_r, itr):
    

    num_pairs = int(down_r.shape[0] * (down_r.shape[0]-1) / 2)
    

    del_ys = np.empty(num_pairs)
    cnt = 0

    same_val = 0
    same_pt = 0

    for i in tqdm(range(down_r.shape[0])):
        for j in range(i+1, down_r.shape[0]):
            pair = [i,j]

            y = fake_f(down_r[pair])
            del_ys[cnt] = y[0] - y[1]

            if (down_r[i] == down_r[j]).any():
                
                print('\n     val: ' + str(down_r[i]) +'\n        : ' + str(down_r[j]))
                same_pt += 1

            if (y[0] - y[1]) == 0:
                print('same val: ' + str(down_r[i]) +'\n        : ' + str(down_r[j]))
                same_val += 1

            cnt += 1

    print('Percentage that evaulate the same' + str(same_val / num_pairs))
    print('Percentage that had at least one that was the same' + str(same_pt / num_pairs))

    n_bins = 20

    y = fake_f(down_r)

    plt.figure()
    plt.hist(y, 50)
    plt.title('ys bins50 itr:' + str(itr))

    plt.figure()
    plt.hist(del_ys, n_bins)
    plt.title('Delta ys bins20 itr:' + str(itr))

    plt.figure()
    plt.hist(del_ys, 100)
    plt.title('Delta ys bins100 itr:' + str(itr))

    
    


def main():
    parser = argparse.ArgumentParser(description='bimodal example with different models and active learners')
    parser.add_argument('--pickle', type=str, default='../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    parser.add_argument('--env', type=int, default=0, help='Environment number')
    parser.add_argument('--fake_func', type=str, default='min', help='Selects fake function type [linear, squared, logistic, sin_exp, min, max, squared_min_max]')
    parser.add_argument('--user', type=str, default='human_choice', help='Selects the type of user [human_choice perfect]')
    parser.add_argument('--p_correct', type=float, default=0.95, help='value to tune synth user to')
    parser.add_argument('--num_alts', type=int, default=2, help='Number of alternative assumed to be passed to user')
    parser.add_argument('--alpha', type=float, default=0.5, help='The alpha to min to logistic function 1 is fully min, 0 is fully logistic')
    args = parser.parse_args()

    dim_rewards = 5

    ###### Open and read path data from Pickle
    with open(args.pickle, 'rb') as f:
        path_data = pickle.load(f)

    num_train = -1
    num_eval = -1

    #train_data = [path_d[:num_train] for path_d in path_data['train']]
    eval_data = [path_d[:num_eval] for path_d in path_data['eval']]
    eval_env_d = eval_data[args.env]

    eval_user_d = np.empty((0, dim_rewards))
    for path_d in eval_env_d:
        eval_user_d = np.append(eval_user_d, path_d['rewards'], axis=0)

    ###### Create fake function and synthetic user
    
    down_r = downsample_kmed_rand(eval_user_d, 50)


    # for i in range(5):
    #     for j in range(i+1, 5):
    #         plt.figure()
    #         plt.title('i: ' + str(i) + ' j: ' + str(j))
    #         plt.scatter(down_r[:,i], down_r[:,j])



    # plt.show()
    # return

    config = {'dim_rewards': dim_rewards, 'alpha_fake': args.alpha}

    for i in range(5):
        fake_f = get_fake_func(args.fake_func, config)
        test_trained_model2(fake_f, down_r, i)

    plt.show()








if __name__ == '__main__':
    main()
