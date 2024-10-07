# Copyright 2024 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
# single_test.py
#
# Written Ian Rankin - January 2024
# A function to start and run a single experiment. 
# Based on single_test_simple I wrote previously

import numpy as np
import argparse
import oyaml as yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

import lop

from experiment_helper import train_and_eval

import sys
import os
import time
import pdb



plan_file_header='./'

train_files =  [['1.plan', '2.plan', '3.plan', '4.plan', '5.plan', '6.plan', '7.plan', '8.plan', '9.plan', '10.plan', '11.plan', '12.plan', '13.plan', '14.plan', '15.plan', '16.plan', '17.plan', '18.plan', '19.plan', '20.plan', ],
                ['31.plan', '32.plan', '33.plan', '34.plan', '35.plan', '36.plan', '37.plan', '38.plan', '39.plan', '40.plan', '41.plan', '42.plan', '43.plan', '44.plan', '45.plan', '46.plan', '47.plan', '48.plan', '49.plan', '50.plan', ],
                ['61.plan', '62.plan', '63.plan', '64.plan', '65.plan', '66.plan', '67.plan', '68.plan', '69.plan', '70.plan', '71.plan', '72.plan', '73.plan', '74.plan', '75.plan', '76.plan', '77.plan', '78.plan', '79.plan', '80.plan', ],
                ['91.plan', '92.plan', '93.plan', '94.plan', '95.plan', '96.plan', '97.plan', '98.plan', '99.plan', '100.plan', '101.plan', '102.plan', '103.plan', '104.plan', '105.plan', '106.plan', '107.plan', '108.plan', '109.plan', '110.plan', ],
                ['121.plan', '122.plan', '123.plan', '124.plan', '125.plan', '126.plan', '127.plan', '128.plan', '129.plan', '130.plan', '131.plan', '132.plan', '133.plan', '134.plan', '135.plan', '136.plan', '137.plan', '138.plan', '139.plan', '140.plan', ],
                ['151.plan', '152.plan', '153.plan', '154.plan', '155.plan', '156.plan', '157.plan', '158.plan', '159.plan', '160.plan', '161.plan', '162.plan', '163.plan', '164.plan', '165.plan', '166.plan', '167.plan', '168.plan', '169.plan', '170.plan', ],
                ['181.plan', '182.plan', '183.plan', '184.plan', '185.plan', '186.plan', '187.plan', '188.plan', '189.plan', '190.plan', '191.plan', '192.plan', '193.plan', '194.plan', '195.plan', '196.plan', '197.plan', '198.plan', '199.plan', '200.plan', ],
                ['211.plan', '212.plan', '213.plan', '214.plan', '215.plan', '216.plan', '217.plan', '218.plan', '219.plan', '220.plan', '221.plan', '222.plan', '223.plan', '224.plan', '225.plan', '226.plan', '227.plan', '228.plan', '229.plan', '230.plan', ],
                ['241.plan', '242.plan', '243.plan', '244.plan', '245.plan', '246.plan', '247.plan', '248.plan', '249.plan', '250.plan', '251.plan', '252.plan', '253.plan', '254.plan', '255.plan', '256.plan', '257.plan', '258.plan', '259.plan', '260.plan', ],
                ['271.plan', '272.plan', '273.plan', '274.plan', '275.plan', '276.plan', '277.plan', '278.plan', '279.plan', '280.plan', '281.plan', '282.plan', '283.plan', '284.plan', '285.plan', '286.plan', '287.plan', '288.plan', '289.plan', '290.plan', ],
                ]



eval_files = [['21.plan', '22.plan', '23.plan', '24.plan', '25.plan', '26.plan', '27.plan', '28.plan', '29.plan', '30.plan', ],
            ['51.plan', '52.plan', '53.plan', '54.plan', '55.plan', '56.plan', '57.plan', '58.plan', '59.plan', '60.plan', ],
            ['81.plan', '82.plan', '83.plan', '84.plan', '85.plan', '86.plan', '87.plan', '88.plan', '89.plan', '90.plan', ],
            ['111.plan', '112.plan', '113.plan', '114.plan', '115.plan', '116.plan', '117.plan', '118.plan', '119.plan', '120.plan', ],
            ['141.plan', '142.plan', '143.plan', '144.plan', '145.plan', '146.plan', '147.plan', '148.plan', '149.plan', '150.plan', ],
            ['171.plan', '172.plan', '173.plan', '174.plan', '175.plan', '176.plan', '177.plan', '178.plan', '179.plan', '180.plan', ],
            ['201.plan', '202.plan', '203.plan', '204.plan', '205.plan', '206.plan', '207.plan', '208.plan', '209.plan', '210.plan', ],
            ['231.plan', '232.plan', '233.plan', '234.plan', '235.plan', '236.plan', '237.plan', '238.plan', '239.plan', '240.plan', ],
            ['261.plan', '262.plan', '263.plan', '264.plan', '265.plan', '266.plan', '267.plan', '268.plan', '269.plan', '270.plan', ],
            ['291.plan', '292.plan', '293.plan', '294.plan', '295.plan', '296.plan', '297.plan', '298.plan', '299.plan', '300.plan', ],
            ]


train_files_short = [ ['1.plan', '2.plan'],
                ['31.plan'],
                ['61.plan'],
                ['91.plan']]

# eval_files = [  ['21.plan', '22.plan'],
#                 ['51.plan', '52.plan'],
#                 ['81.plan', '82.plan'],
#                 ['111.plan', '112.plan']]
eval_files_short = [  ['21.plan', '22,plan'],
                ['51.plan'],
                ['81.plan'],
                ['111.plan']]


## str_timestamp
# creates a string versiion of a timestamp.
# Designed to be used to save files without having to worry about conflicts
# between runs.
#
# @return the timestamp as a string.
def str_timestamp():
    t = time.localtime()
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', t)
    return timestamp

possible_selectors = ['UCB', 'SGV_UCB', 'RANDOM', 'MUTUAL_INFO', 'MUTUAL_INFO_PERF', \
                        'BAYES_INFO_GAIN_PROBIT', 'BAYES_INFO_GAIN_999', \
                        'ACQ_RHO', 'ACQ_LL', 'ACQ_EPIC', 'ACQ_SPEAR',\
                        'SW_BAYES_PROBIT', 'SW_ACQ_RHO', 'SW_ACQ_EPIC', 'SW_ACQ_LL', 'SW_ACQ_SPEAR',\
                        'SW_UCB_RHO', 'SW_UCB_EPIC', 'SW_UCB_LL', 'SW_UCB_SPEAR', \
                        'SW_FIXED_RHO', 'SW_FIXED_EPIC', 'SW_FIXED_LL', 'SW_FIXED_SPEAR', \
                        'ABS_ACQ_RHO', 'ABS_ACQ_LL', 'ABS_ACQ_EPIC', 'ABS_ACQ_SPEAR',\
                        'ABS_BAYES_PROBIT']
possible_selection_types = ['choose1', 'ranking', 'rating', 'switch']
possible_fake_funcs = ['linear', 'squared', 'logistic', 'sin_exp', 'max', 'min', 'squared_min_max']
possible_models = ['gp', 'linear']

def boolean_string(s):
    if s not in {'False', 'True', 'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True' or s == 'true'

def main():
    # Read in arguments required
    parser = argparse.ArgumentParser(description='User preferences')
    parser.add_argument('--config', type=str, default='./config.yaml')
    parser.add_argument('--dir', type=str, default='')
    parser.add_argument('--env', type=int, default=2, help='environment number, using 2,4,7 [0,9] allowed')
    parser.add_argument('--model', type=str, default='gp', help='Set the model to '+str(possible_models))
    parser.add_argument('--selector', type=str, default='SGV_UCB', help='Set the selectors to use options '+str(possible_selectors))
    parser.add_argument('--sel_type', type=str, default='choose1', help='Set the selection type to use options '+str(possible_selection_types))
    parser.add_argument('--num_runs', type=int, default=100, help='The number of runs')
    parser.add_argument('--num_alts', type=int, default=4, help='The number of alternatives to show to the user')
    parser.add_argument('--user', type=str, default='human_choice', help='Set the synthetic user type to use options [ perfect human_choice ]')
    parser.add_argument('--hyper', type=str, default='hyper', help='Sets whether to perform hyperparameter optimization [ hyper no ]')
    parser.add_argument('--def_pareto', type=str, default='false', help='Sets whether optimization defaults to user pareto optimal points first or not bool [true, false]')
    parser.add_argument('--fake_func', type=str, default='linear', help='fake function for synthetic user: '+str(possible_fake_funcs))
    parser.add_argument('--test_experiment', type=bool, default=False, help='Shortens the number of plans to make testing the experiment easier')
    parser.add_argument('--kmedoid', type=str, default='medrand', help='Sets whether to use kmediod downsampling to decrease the input points to the query selection algorithm [true, uni, downhull, medrand]')
    parser.add_argument('--v_abs', type=float, default=80.0, help='abs probit v parameter default=80.0')
    parser.add_argument('--sigma_abs', type=float, default=1.0, help='abs probit sigma parameter default=1.0')
    parser.add_argument('--sigma_pair', type=float, default=1.0, help='abs probit sigma parameter default=1.0')
    parser.add_argument('--rbf_sigma', type=float, default=1.0, help='RBF sigma parameter (unused for linear) default=1.0')
    parser.add_argument('--rbf_l', type=float, default=0.4, help='lengthscale of the rbf, unused for linear default=0.4')
    parser.add_argument('--p_synth_pair', type=float, default=0.95, help='tuned average probability of selecting correct synth pair')
    parser.add_argument('--p_synth_abs', type=float, default=0.95, help='Tuned average probability of ratings matching the correct pair similar to the synth pair abs')
    parser.add_argument('-v', type=bool, default=False, help='Verbose print statements')
    args = parser.parse_args()

    global train_files
    global eval_files
    if args.test_experiment:
        train_files = train_files_short
        eval_files = eval_files_short
    train_files = [[plan_header+file for file in env] for env in train_files]
    eval_files = [[plan_header+file for file in env] for env in eval_files]

    if args.selector not in possible_selectors:
        print('Selector should be one of these '+str(possible_selectors)+' not ' + str(args.selector))
        sys.exit(0)
    if args.fake_func not in possible_fake_funcs:
        print('fake_func should be one of these '+str(possible_fake_funcs)+' not ' + str(args.fake_func))
        sys.exit(0)
    if args.model not in possible_models:
        print('model should be one of these '+str(possible_models)+' not ' + str(args.model))
        sys.exit(0)
    if args.sel_type not in possible_selection_types:
        print('selection types should be one of these '+str(possible_selection_types)+' not ' + str(args.sel_type))
        sys.exit(0)

    if args.def_pareto == 'true':
        args.def_pareto = True
    elif args.def_pareto == 'false':
        args.def_pareto = False
    else:
        print('def_pareto should be either [true, false]')
        sys.exit(0)

    # create a results folder named by the selector, user type, fake_func, and environment number.
    if args.dir == '':
        folder_name = 'results/AT_'+args.selector+'_model_'+args.model+'_'+args.sel_type+'_user_'+args.user+str(args.num_alts)+'_fake_'+args.fake_func+'_pareto_' + str(args.def_pareto) + '_kmed_' +str(args.kmedoid)+ '_ppair_' + str(args.p_synth_pair) + '_pabs_' + str(args.p_synth_abs) + '_'+args.hyper+'_v_'+str(args.v_abs) + '_sigabs_' + str(args.sigma_abs) + '_sigpair_'+str(args.sigma_pair)+'_rbfl_'+str(args.rbf_l)+'_rbfsig_'+str(args.rbf_sigma)+'_env'+str(args.env)+'_'+str_timestamp()+'/'
    else:
        folder_name = 'results/'+args.dir




    os.mkdir(folder_name)

    num_runs = args.num_runs
    num_eval = len(eval_files[0])
    num_train = len(train_files[0])
    #num_train = 10

    # setup record data.
    ranks = np.empty((num_runs, num_train+1, num_eval))
    estimated_scores = np.empty((num_runs, num_train+1, num_eval))
    real_scores = np.empty((num_runs, num_train+1, num_eval))
    score_diff = np.empty((num_runs, num_train+1, num_eval))
    spearmans = np.empty((num_runs, num_train+1, num_eval))
    pearsons = np.empty((num_runs, num_train+1, num_eval))
    rhos = np.empty((num_runs, num_train+1, num_eval))
    avg_ranks = np.empty((num_runs, num_train+1))
    avg_correct = np.empty((num_runs, num_train+1))
    query_type_is_abs = np.empty((num_runs, num_train+1))

    cur_run = 0
    try:
        path_data = None
        for j in tqdm(range(num_runs)):
            cur_run = j
            run_folder = folder_name+'run_'+str(j)+'/'
            os.mkdir(run_folder)
            print('VERBOSE: ' + str(args.v))
            accuracy, avg_selection, all_ranks, est_score, real_score, s_diff, query_type, spearman, pearson, rho, path_data = \
                            train_and_eval( args.config, \
                                            env_num=args.env, \
                                            fake_function_desc=args.fake_func, \
                                            folder=run_folder, \
                                            selector=args.selector, \
                                            selection_type=args.sel_type,
                                            model_desc=args.model, \
                                            num_alts = args.num_alts, \
                                            default_pareto = args.def_pareto, \
                                            synth_user=args.user, \
                                            hyper=args.hyper, \
                                            num_train=num_train, \
                                            num_eval=num_eval,\
                                            sigma_abs=args.sigma_abs,\
                                            sigma_pair=args.sigma_pair,\
                                            v=args.v_abs,\
                                            rbf_l=args.rbf_l,\
                                            rbf_sigma=args.rbf_sigma, \
                                            use_kmedoid=args.kmedoid, \
                                            p_synth_pair=args.p_synth_pair, \
                                            p_synth_abs=args.p_synth_abs, \
                                            path_data = path_data, \
                                            verbose = args.v)


            avg_correct[j] = accuracy
            avg_ranks[j] = avg_selection
            query_type_is_abs[j] = query_type
            
            ranks[j] = all_ranks

            estimated_scores[j] = est_score
            real_scores[j] = real_score
            spearmans[j] = spearman
            pearsons[j] = pearson
            rhos[j] = rho
            score_diff[j] = s_diff

            print('Updating savez file with data from run: '+str(j))
            np.savez(folder_name+'train_data', 
                    avg_correct=avg_correct[:j+1], 
                    avg_ranks=avg_ranks[:j+1], \
                    ranks=ranks[:j+1], 
                    estimated_scores=estimated_scores[:j+1], \
                    real_scores = real_scores[:j+1], \
                    score_diff = score_diff[:j+1], \
                    spearmans = spearmans[:j+1], \
                    pearsons = pearsons[:j+1], \
                    rhos = rhos[:j+1], \
                    query_type_is_abs = query_type_is_abs[:j+1])

    except KeyboardInterrupt:
        j = cur_run

        np.savez(folder_name+'train_data', 
                    avg_correct=avg_correct[:j+1], 
                    avg_ranks=avg_ranks[:j+1], \
                    ranks=ranks[:j+1], 
                    estimated_scores=estimated_scores[:j+1], \
                    real_scores = real_scores[:j+1], \
                    spearmans = spearmans[:j+1], \
                    pearsons = pearsons[:j+1], \
                    rhos = rhos[:j+1], \
                    score_diff = score_diff[:j+1], \
                    query_type_is_abs = query_type_is_abs[:j+1])
        sys.exit(0)

if __name__ == '__main__':
    plan_header = plan_file_header+'saved_plans/'

    main()

