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


def test_trained_model(user_f, eval_r):
    print(eval_r)
    print(user_f)
    #pdb.set_trace()

    print('Rating')

    down_r = downsample_kmed_rand(eval_r, 50)

    num_pairs = int(down_r.shape[0] * (down_r.shape[0]-1) / 2)
    
    rates = np.empty(num_pairs)
    chooses = np.empty(num_pairs)
    del_ys = np.empty(num_pairs)

    cnt = 0
    for i in tqdm(range(down_r.shape[0])):
        for j in range(i+1, down_r.shape[0]):
            pair = [i,j]

            rate_avg, choose_avg, delta_y =  test_pair_error_model(user_f, eval_r, pair)
            rates[cnt] = rate_avg
            chooses[cnt] = choose_avg
            del_ys[cnt] = delta_y

            cnt += 1

    n_bins = 200
    plt.hist(rates,n_bins)
    plt.title('Rate probability')

    plt.figure()
    plt.hist(chooses, n_bins)
    plt.title('Choose probabilties')

    plt.figure()
    plt.hist(del_ys, n_bins)
    plt.title('Delta ys')

    plt.show()
        

def test_trained_model2(user_f, eval_r):
    down_r = downsample_kmed_rand(eval_r, 50)

    num_pairs = int(down_r.shape[0] * (down_r.shape[0]-1) / 2)
    

    del_ys = np.empty(num_pairs)
    cnt = 0
    for i in tqdm(range(down_r.shape[0])):
        for j in range(i+1, down_r.shape[0]):
            pair = [i,j]

            y = user_f.fake_f(eval_r[pair])
            del_ys[cnt] = y[0] - y[1]

            cnt += 1

    n_bins = 20

    plt.figure()
    plt.hist(del_ys, n_bins)
    plt.title('Delta ys')

    plt.figure()
    plt.hist(del_ys, 50)
    plt.title('Delta ys')

    plt.show()
    



def test_pair_error_model(user_f, eval_r, pair):
    N=1000
    correct = 0

    y = user_f.fake_f(eval_r[pair])
    delta_y = y[0] - y[1]
    for i in range(N):
        r = user_f.rate(eval_r[pair])
        

        #print(r)

        if (r[0] > r[1]) == (y[0] > y[1]):
            correct += 1

    rate_avg = correct / N
        
    correct = 0
    for i in range(N):
        r = user_f.choose(eval_r[pair])

        #print(r)

        if r == np.argmax(y):
            correct += 1

    choose_avg = correct / N

    return rate_avg, choose_avg, delta_y



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

    np.random.seed(0)
    random.seed(0)

    for i in range(1):
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
    #test_trained_model(user_f, eval_user_d)
    test_trained_model2(user_f, eval_user_d)










if __name__ == '__main__':
    main()
