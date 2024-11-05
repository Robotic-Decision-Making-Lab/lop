# intermediate_loading.py
# Written Ian Rankin - Novemeber 2024
#
# code to read in experiment files and load the GP model from the state they are in.

import lop

import numpy as np
import glob
from datetime import datetime as dt
import pickle
from sklearn_extra.cluster import KMedoids
import pdb


def filter_by_time(names, after=None, before=None):
    fmt_str = '%Y-%m-%d_%H-%M-%S'
    
    if after is not None:
        after_d = dt.strptime(after, fmt_str)
        
        if before is not None:
            # only keep values between before and after
            
            before_d = dt.strptime(before, fmt_str)
            filtered = [n for n in names if dt.strptime(n.split('/')[5][-19:], fmt_str) > after_d and dt.strptime(n.split('/')[5][-19:], fmt_str) < before_d]
        else:
            # only keep values after
            filtered = [n for n in names if dt.strptime(n[-34:-15], fmt_str) > after_d]
    else:
        if before is not None:
            # only keep names before
            before_d = dt.strptime(before, fmt_str)
            filtered = [n for n in names if dt.strptime(n[-34:-15], fmt_str) < before_d]
        else:
            # just return everything
            filtered = names
    return filtered


def get_filepath(env=4, run=14, trainitr=15, fake_f_name='logistic', p_abs='0.85', selector='SW_ACQ_RHO', idx=0):
    #env=4
    #run=14
    #trainitr=15

    sel = selector
    typ = 'switch'
    if selector == 'ABS_UCB':
        sel = 'UCB'
        typ = 'rating'

    #glob_str = '../experiments/results/AT_SW_UCB_RHO_model_gp_switch_user_human_choice2_fake_'+fake_f_name+'_pareto_False_kmed_medrand_ppair_0.95_pabs_0.95_no_v_60.0_sigabs_1.0_sigpair_0.1_rbfl_1.2_rbfsig_1.0_env'+str(env)+'_**/run_'+str(run)+'/trainitr_'+str(trainitr)+'_viz.npz'
    glob_str = '../../../experiments/results/AT_'+sel+'_model_gp_'+typ+'_user_human_choice2_fake_'+fake_f_name+'_pareto_False_kmed_medrand**_pabs_'+p_abs+'**_no_v_60.0_sigabs_1.0_sigpair_0.1**_rbfl_1.2_rbfsig_1.0_env'+str(env)+'_**/run_'+str(run)+'/trainitr_'+str(trainitr)+'_viz.npz'
    
    print('glob_str: ' + glob_str)
    possible = glob.glob(glob_str)
    print(possible)
    possible = filter_by_time(possible, after='2024-10-28_00-00-00', before='2025-10-22_00-00-00')
    if idx == -1:
        return possible
    filepath = possible[0]
    return filepath


def model_load(model, filepath):
    data = np.load(filepath, allow_pickle=True)
    
    
    
    model.X_train = data['GP_pts']
    
    if len(data['GP_pref_0'].shape) > 0:
        model.y_train[0] = data['GP_pref_0']
    else:
        model.y_train[0] = None
        
    model.y_train[1] = None

    if len(data['GP_pref_2'].shape) > 0:
        model.y_train[2] = [data['GP_pref_2'][0], data['GP_pref_2'][1].astype(int)]
    else:
        model.y_train[2] = None
    model.prior_idx = data['GP_prior_idx']
    model.optimized = False



def downsample_kmed_rand(X, num_downselect, kmed_pct=0.5):
    num_kmed = int(kmed_pct * num_downselect)
    num_rand = num_downselect - num_kmed
    
    rand_idxs = np.random.choice(X.shape[0], num_rand, replace=False)
    non_rand_idxs = list(set(list(range(X.shape[0]))) - set(rand_idxs))
    
    cent_pts = KMedoids(n_clusters=num_kmed).fit(X[non_rand_idxs]).cluster_centers_

    return np.append(cent_pts, X[rand_idxs], axis=0)


def main():
    parser = argparse.ArgumentParser(description='Load a model at an intermediate point')
    parser.add_argument('--env', type=int, default=0, help='Environment number')
    parser.add_argument('--pickle', type=str, default='../../../../other_repos/ice-soil-planner/scripts/plan_saving/rewards_ice_soil.pickle', help='Set the location of the pickle file to load')
    parser.add_argument('--trainitr', type=int, default=2, help='Train iteration')
    parser.add_argument('--p_abs', type=str, default='0.7', help='p_abs')
    args = parser.parse_args()


    env=args.env
    trainitr=args.trainitr
    p_abs = args.p_abs


    ###### Open and read path data from Pickle
    with open(args.pickle, 'rb') as f:
        path_data = pickle.load(f)
    
    train_data = [path_d[::] for path_d in path_data['train']]
    eval_data = [path_d[::] for path_d in path_data['eval']]

    eval_rew = [[env_p['rewards'] for env_p in data_env] for data_env in eval_data]
    train_rew = [[env_p['rewards'] for env_p in data_env] for data_env in train_data]

    eval_rew_example = eval_rew[env][np.random.randint(0,10)]
    candidate_pts = downsample_kmed_rand(eval_rew_example, 50)

    M=200
    #abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman')
    pair = lop.AcquisitionSelection(M=M, alignment_f='spearman')
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


    model_load(model, get_filepath(trainitr=trainitr,env=env, fake_f_name='min', p_abs=p_abs, selector='SW_ALT_SPEAR'))

    model.active_learner.num_calls = trainitr
    #model.select(np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0,0,0,0,0]]), 2)
    model.select(candidate_pts, 2)

    print('Done with selecting')



if __name__ == '__main__':
    import argparse
    main()
