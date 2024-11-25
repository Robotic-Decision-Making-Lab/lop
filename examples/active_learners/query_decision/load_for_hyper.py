# load_for_hyper.py
# Written Ian Rankin - November 2024
#
# 

import lop

import numpy as np
import glob
from datetime import datetime as dt
import pickle
from sklearn_extra.cluster import KMedoids
import argparse
import pdb

from intermediate_loading import filter_by_time, get_filepath, model_load, downsample_kmed_rand

def main():
    parser = argparse.ArgumentParser(description='Load a model at an intermediate point')
    parser.add_argument('--env', type=int, default=0, help='Environment number')
    parser.add_argument('--pickle', type=str, default='../../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    parser.add_argument('--trainitr', type=int, default=2, help='Train iteration')
    parser.add_argument('--p_abs', type=str, default='0.95', help='p_abs')
    parser.add_argument('--run', type=int, default=0, help='run number')
    args = parser.parse_args()


    env=args.env
    trainitr=args.trainitr
    p_abs = args.p_abs
    run = args.run

    ###### Open and read path data from Pickle
    with open(args.pickle, 'rb') as f:
        path_data = pickle.load(f)
    
    train_data = [path_d[::] for path_d in path_data['train']]
    eval_data = [path_d[::] for path_d in path_data['eval']]

    eval_rew = [[env_p['rewards'] for env_p in data_env] for data_env in eval_data]
    train_rew = [[env_p['rewards'] for env_p in data_env] for data_env in train_data]

    eval_rew_example = eval_rew[env][np.random.randint(0,10)]
    candidate_pts = downsample_kmed_rand(eval_rew_example, 50)

    M=300
    #abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman')
    pair = lop.AcquisitionSelection(M=M, alignment_f='spearman', rep_Q_method='stable',
                            rep_Q_data={'filename': '../../../experiments/comparision_pts.npy'})
    abs_l = lop.UCBLearner(1.0)
    al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l, num_calls_decision=trainitr,
                                default_to_pareto=False, always_select_best=False) 


    #active_learner = None
    rbf_sigma=1.0
    rbf_l = 1.2

    model = lop.PreferenceGP(
                        cov_func=lop.RBF_kern(rbf_sigma, rbf_l, sigma_noise=0.00001),
                        normalize_gp=False,
                        pareto_pairs=False,
                        normalize_positive=False,
                        use_hyper_optimization=False,
                        active_learner= al)


    model.probits[0].set_sigma(0.1)
    model.probits[2].set_sigma(1.0)
    model.probits[2].set_v(60.0)


    for i in range(1):

        model_load(model, get_filepath(trainitr=trainitr,env=env, run=run, fake_f_name='min', p_abs=p_abs, selector='SW_ALT_SPEAR'))

        model.active_learner.num_calls = trainitr
        #model.select(np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0,0,0,0,0]]), 2)
        
        model.optimize(optimize_hyperparameter=True)

        print(model.get_hyper())
        print('\n\n')

    

    print('Done with selecting')



if __name__ == '__main__':
    main()

