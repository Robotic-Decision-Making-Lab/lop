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
    #parser.add_argument('--p_abs', type=float, default=0.95, help='value to tune synth user to for ratings')
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

    usr.learn_beta(eval_rew, 0.95, Q_size=2, p_sigma=0.7)



    


if __name__ == '__main__':
    main()
