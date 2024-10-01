# synth_user_example.py
# Written Ian Rankin - September 2024
#
# A set of code to show an example usage of the synthetic user class.
# Additionally, it allows testing the class with different examples


import numpy as np
import pickle
import argparse
import lop

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../../experiments')

from experiment_helper import get_synth_user, get_fake_func



def main():
    parser = argparse.ArgumentParser(description='bimodal example with different models and active learners')
    parser.add_argument('--pickle', type=str, default='../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    parser.add_argument('--env', type=int, default=0, help='Environment number')
    parser.add_argument('--fake_func', type=str, default='min', help='Selects fake function type [linear, squared, logistic, sin_exp, min, max, squared_min_max]')
    parser.add_argument('--user', type=str, default='human_choice', help='Selects the type of user [human_choice perfect]')
    parser.add_argument('--p_correct', type=float, default=0.95, help='value to tune synth user to')
    parser.add_argument('--num_alts', type=int, default=2, help='Number of alternative assumed to be passed to user')
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
    
    config = {'dim_rewards': dim_rewards}


    for i in range(10):
        print('Starting run '+str(i)+' of tuning synth user')
        fake_f = get_fake_func(args.fake_func, config)
        user_f = get_synth_user(args.user, fake_f, config)

        test_pts = np.array([[0,0,0,0,0], [1,1,1,1,1], 
                             [1.05, 1.05, 1.05, 1.05, 1.05],
                             [1.1, 1.1, 1.1, 1.1, 1.1],
                             [2,2,2,2,2]])
        y_f = fake_f(test_pts)
        print(y_f)
        print(fake_f)

        user_f.learn_beta(eval_user_d, args.p_correct, Q_size=args.num_alts)
        print('user_f sigma: ' + str(user_f.sigma) +' beta: ' + str(user_f.beta))
        print('\n\n')


    print(user_f)










if __name__ == '__main__':
    main()
