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
    parser = argparse.ArgumentParser(description='IDK, figure it out')
    parser.add_argument('--pickle', type=str, default='../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    parser.add_argument('--fake_func', type=str, default='min', help='Selects fake function type [linear, squared, logistic, sin_exp, min, max, squared_min_max]')
    parser.add_argument('--user', type=str, default='human_choice', help='Selects the type of user [human_choice perfect]')
    parser.add_argument('--p_pair', type=float, default=0.95, help='value to tune synth user to for choose comparisions')
    parser.add_argument('--p_abs', type=float, default=0.95, help='value to tune synth user to for ratings')
    parser.add_argument('--num_alts', type=int, default=2, help='Number of alternative assumed to be passed to user')
    args = parser.parse_args()

    dim_rewards = 5

    ###### Open and read path data from Pickle
    with open(args.pickle, 'rb') as f:
        path_data = pickle.load(f)
    
    train_data = [path_d[::] for path_d in path_data['train']]
    eval_data = [path_d[::] for path_d in path_data['eval']]

    eval_rew = [[env_p['rewards'] for env_p in data_env] for data_env in eval_data]
    train_rew = [[env_p['rewards'] for env_p in data_env] for data_env in train_data]


    # Finish loading data!
    envs = [0,1,2,3,4,5,6,7,8,9]
    
    f = lop.FakeWeightedMin(dim_rewards)
    usr = lop.HumanChoiceUser2(f)

    usr.learn_beta(eval_rew, args.p_pair, Q_size=2, p_sigma=args.p_abs)

    ### Test user synth

    N = 10000

    train_rew = eval_rew

    count_choose = 0
    count_rate = 0
    total_count = 0

    for i in range(N):
        env_num = int(i * (len(train_rew) / N))
        pro_num = np.random.randint(0, len(train_rew[env_num]))
        pair = np.random.choice(train_rew[env_num][pro_num].shape[0], 2, replace=False)

        rew_pair = train_rew[env_num][pro_num][pair]
        idx = usr.choose(rew_pair)
        idx_rate = np.argmax(usr.rate(rew_pair))
        score_pair = usr.fake_f(rew_pair)
        corr_idx = np.argmax(score_pair)

        if score_pair[0] != score_pair[1]:
            if corr_idx == idx:
                count_choose += 1

            if idx_rate == idx:
                count_rate += 1 
            total_count += 1

    acc_rate = count_rate / total_count
    acc = count_choose / total_count

    print('All acc choose = ' + str(acc))
    print('All acc rate = ' + str(acc_rate))

    N = 7500

    count_choose = 0
    count_rate = 0
    total_count = 0

    for i in range(N):
        #env_num = int(i * (len(train_rew) / N))
        env_num = 7
        pro_num = np.random.randint(0, len(train_rew[env_num]))
        pair = np.random.choice(train_rew[env_num][pro_num].shape[0], 2, replace=False)

        rew_pair = train_rew[env_num][pro_num][pair]
        idx = usr.choose(rew_pair)
        idx_rate = np.argmax(usr.rate(rew_pair))
        score_pair = usr.fake_f(rew_pair)
        corr_idx = np.argmax(score_pair)

        if score_pair[0] != score_pair[1]:
            if corr_idx == idx:
                count_choose += 1

            if idx_rate == idx:
                count_rate += 1 
            total_count += 1

    acc_rate = count_rate / total_count
    acc = count_choose / total_count

    print('Env acc choose = ' + str(acc))
    print('Env acc rate = ' + str(acc_rate))




    


if __name__ == '__main__':
    main()
