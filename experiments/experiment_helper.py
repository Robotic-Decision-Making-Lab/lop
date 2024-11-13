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

# test_helper.py
# Written Ian Rankin - January 2024
#
# A set of functions to setup and perform the active learning training and eval.

import numpy as np
import oyaml as yaml
import sys
import pickle
from sklearn_extra.cluster import KMedoids
import scipy.spatial as spa
import pdb

import lop

from GP_visualization import record_gp_state, visualize_data, visualize_single_run_regret

def get_active_learner(selector, selection_type, rep_pt_type, UCB_scalar, default_to_pareto, config, fake_func=None):
    #default_to_pareto = config['default_to_pareto']
    if selection_type == 'rating':
        always_select_best = False
    else:
        always_select_best = config['always_select_best']

    M=200
    alpha = config['alpha']

    rep_type = rep_pt_type# ['stable' 'sampled']
    if rep_type == 'sampled':
        rep_pt_data = {'num_pts': 30, 'num_Q': 50}
    elif rep_type == 'stable':
        rep_pt_data = {'filename': config['stable_comp_file']}
    


    al = None
    if selector == 'UCB':
        al = lop.UCBLearner(UCB_scalar, default_to_pareto, always_select_best)
    elif selector == 'SGV_UCB':
        al = lop.GV_UCBLearner(UCB_scalar, default_to_pareto, always_select_best)
    elif selector == 'MUTUAL_INFO':
        al = lop.MutualInfoLearner(None, default_to_pareto, always_select_best)
    elif selector == 'MUTUAL_INFO':
        al = lop.MutualInfoLearner(None, default_to_pareto, always_select_best)
    elif selector == 'MUTUAL_INFO_PERF':
        al = lop.MutualInfoLearner(fake_func, default_to_pareto, always_select_best)
    elif selector == 'RANDOM':
        al = lop.RandomLearner(default_to_pareto, always_select_best)
    elif selector == 'BAYES_INFO_GAIN_PROBIT':
        al = lop.BayesInfoGain(default_to_pareto, always_select_best,p_q_B_method='probit')
    elif selector == 'ABS_BAYES_PROBIT':
        al = lop.BayesInfoGain(default_to_pareto, always_select_best,p_q_B_method='probit')
    elif selector == 'BAYES_INFO_GAIN_999':
        al = lop.BayesInfoGain(default_to_pareto, always_select_best,p_q_B_method='999')
    elif selector == 'ACQ_RHO':
        al = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'ACQ_LL':
        al = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)   
    elif selector == 'ACQ_EPIC':
        al = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'ACQ_SPEAR':
        al = lop.AcquisitionSelection(M=M, alignment_f='spearman',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'ABS_ACQ_RHO':
        al = lop.AbsAcquisition(M=M, alignment_f='rho',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'ABS_ACQ_LL':
        al = lop.AbsAcquisition(M=M, alignment_f='loglikelihood',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)   
    elif selector == 'ABS_ACQ_EPIC':
        al = lop.AbsAcquisition(M=M, alignment_f='epic',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'ABS_ACQ_SPEAR':
        al = lop.AbsAcquisition(M=M, alignment_f='spearman',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best,
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
    elif selector == 'SW_ACQ_RHO':
        abs_l = lop.AbsAcquisition(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_ACQ_LL':
        abs_l = lop.AbsAcquisition(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_ACQ_EPIC':
        abs_l = lop.AbsAcquisition(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_ACQ_SPEAR':
        abs_l = lop.AbsAcquisition(M=M, alignment_f='spearman')
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman')
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_BAYES_PROBIT':
        abs_l = lop.AbsBayesInfo(M=M)
        pair = lop.BayesInfoGain()
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_UCB_RHO':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparision(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best, alpha=alpha)              
    elif selector == 'SW_UCB_LL':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparision(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best, alpha=alpha)   
    elif selector == 'SW_UCB_SPEAR':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman')
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman')
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparision(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best, alpha=alpha)
    elif selector == 'SW_UCB_EPIC':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='epic')
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic')
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparision(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best, alpha=alpha)   
    elif selector == 'SW_FIXED_RHO':
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionSetFixed(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_FIXED_LL':
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionSetFixed(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_FIXED_SPEAR':
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionSetFixed(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_FIXED_EPIC':
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionSetFixed(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_ALT_RHO':
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedAlternating(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_ALT_LL':
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedAlternating(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_ALT_SPEAR':
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedAlternating(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_ALT_EPIC':
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedAlternating(pairwise_l=pair, abs_l=abs_l, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_DEC_RHO':
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l,num_calls_decision=2, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_DEC_LL':
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood')
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l,num_calls_decision=2, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_DEC_SPEAR':
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l,num_calls_decision=2, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_DEC_EPIC':
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l,num_calls_decision=2, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'SW_CHECK_RHO':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='rho',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionEqualChecking(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_CHECK_LL':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='loglikelihood',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionEqualChecking(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_CHECK_SPEAR':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='spearman',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionEqualChecking(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best) 
    elif selector == 'SW_CHECK_EPIC':
        abs_comp = lop.AbsAcquisition(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        pair = lop.AcquisitionSelection(M=M, alignment_f='epic',
                                    rep_Q_method = rep_type, rep_Q_data = rep_pt_data)
        abs_l = lop.UCBLearner(UCB_scalar)
        al = lop.MixedComparisionEqualChecking(pairwise_l=pair, abs_l=abs_l, abs_comp=abs_comp, default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)  


    return al

def get_model(model_desc, active_learner, hyper, config):
    pareto_pairs = config['pareto_pairs']


    model = None
    if model_desc == 'gp':
        if hyper == 'no':
            use_hyper = False
        elif hyper == 'hyper':
            use_hyper = True
        else:
            print('Bad hyperparameter setting in get_model ' + str(hyper))
            sys.exit(0)

        model = lop.PreferenceGP(
                    cov_func=lop.RBF_kern(config['rbf_sigma'], config['rbf_lengthscale'], sigma_noise=config['sigma_noise']),
                    normalize_gp=config['normalize_gp'],
                    pareto_pairs=pareto_pairs,
                    normalize_positive=config['normalize_postive'],
                    use_hyper_optimization=use_hyper,
                    active_learner= active_learner
        )
    elif model_desc == 'linear':
        model = lop.PreferenceLinear(pareto_pairs=pareto_pairs,\
                                    active_learner=active_learner)
    
    model.probits[0].set_sigma(config['sigma_pair'])
    model.probits[2].set_sigma(config['sigma_abs'])
    model.probits[2].set_v(config['v'])

    return model

def get_synth_user(user_desc, utility_f, config):
    if user_desc == 'perfect':
        func = lop.PerfectUser(utility_f)
    elif user_desc == 'human_choice':
        func = lop.HumanChoiceUser2(utility_f)

    return func

def get_fake_func(fake_func_desc, config):
    if fake_func_desc == 'linear':
        func = lop.FakeLinear(config['dim_rewards'])
    elif fake_func_desc == 'squared':
        func = lop.FakeSquared(config['dim_rewards'])
    elif fake_func_desc == 'logistic':
        func = lop.FakeLogistic(config['dim_rewards'])
    elif fake_func_desc == 'sin_exp':
        func = lop.FakeSinExp(config['dim_rewards'])
    elif fake_func_desc == 'min':
        func = lop.FakeWeightedMin(config['dim_rewards'])
    elif fake_func_desc == 'max':
        func = lop.FakeWeightedMax(config['dim_rewards']) 
    elif fake_func_desc == 'squared_min_max':
        func = lop.FakeSquaredMinMax(config['dim_rewards'])
    elif fake_func_desc == 'min_log':
        func = lop.FakeMinLog(config['dim_rewards'], config['alpha_fake'])

    return func

# Linear [0.17191523855986895, 2.41264845341967]
# Multiply (0.0208195, 5.8210312)
# F [550.2325553878064, 1.3174585554818629e+32]
rating_bounds_global = [0.17191523855986895, 2.41264845341967]
def rating_score_from_fake_f(rewards, fake_func, rating_bounds):
    decision_pts = np.arange(rating_bounds[0], rating_bounds[1], (rating_bounds[1] - rating_bounds[0]) / 5)
    decision_pts[0] = 0
    real_y = fake_func(rewards[np.newaxis,:])


    ratings = []
    for y in real_y:
        idx = -1
        for i, pt in enumerate(decision_pts):
            print('y: ' + str(y) + ' pt: ' + str(pt))
            if y > pt:
                idx = i

        ratings.append(idx/5.0 + 0.1)

    return ratings

def downsample_hull(X, num_downselect):
    hull_idxs = spa.ConvexHull(X).vertices

    if len(hull_idxs) < num_downselect:

        non_hull_idxs = list(set(list(range(X.shape[0]))) - set(hull_idxs))

        cents_non_hull = KMedoids(n_clusters=num_downselect - len(hull_idxs)).fit(X[non_hull_idxs]).cluster_centers_


        down_pts = np.append(X[hull_idxs], cents_non_hull, axis=0)
    else:
        print(len(X[hull_idxs]))
        cents_hulls = KMedoids(n_clusters=num_downselect).fit(X[hull_idxs]).cluster_centers_

        down_pts = cents_hulls
    return down_pts

def downsample_kmed_rand(X, num_downselect, kmed_pct=0.5):
    num_kmed = int(kmed_pct * num_downselect)
    num_rand = num_downselect - num_kmed
    
    rand_idxs = np.random.choice(X.shape[0], num_rand, replace=False)
    non_rand_idxs = list(set(list(range(X.shape[0]))) - set(rand_idxs))
    
    cent_pts = KMedoids(n_clusters=num_kmed).fit(X[non_rand_idxs]).cluster_centers_

    return np.append(cent_pts, X[rand_idxs], axis=0)


PAIR_QUERY = 0
ABS_QUERY = 1

def train_and_eval(config_filename, 
                    env_num, 
                    folder,
                    selector='UCB',
                    selection_type='choose1',
                    rep_pt_type='stable',
                    model_desc='gp',
                    fake_function_desc='linear',
                    num_alts=4,
                    UCB_scaler=None,
                    rbf_sigma=None,
                    synth_user='perfect',
                    hyper='no',
                    default_pareto=False,
                    num_train=10,
                    num_eval=10,
                    sigma_abs=0.1,
                    sigma_pair=1.0,
                    v=80.0,
                    use_kmedoid=True,
                    alpha=0.5,
                    alpha_fake=0.5,
                    p_synth_pair=0.95,
                    p_synth_abs=0.95,
                    rbf_l=None,
                    path_data=None,
                    verbose = False):
    #
    with open(config_filename, 'rb') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    print('Staring to read in pickle file')
    if path_data is None:
        with open(config['simple_rewards_file'], 'rb') as f:
            path_data = pickle.load(f)

    print('Done reading pickle file')

    train_data = [path_d[:num_train] for path_d in path_data['train']]
    eval_data = [path_d[:num_eval] for path_d in path_data['eval']]

    dim_rewards = eval_data[env_num][0]['rewards'].shape[1]

    if rbf_sigma is not None:
        config['rbf_sigma'] = rbf_sigma
    if rbf_l is not None:
        config['rbf_lengthscale'] = rbf_l
    if UCB_scaler is None:
        UCB_scaler = config['UCB_scalar']

    num_training = num_train#10

    config['alpha'] = alpha

    #### Get required models and fake functions
    active_learner = get_active_learner(selector, selection_type, rep_pt_type, UCB_scaler, default_pareto, config)
    config['sigma_pair'] = sigma_pair
    config['sigma_abs'] = sigma_abs
    config['v'] = v
    config['dim_rewards'] = dim_rewards
    config['alpha_fake'] = alpha_fake
    # config['rbf_lengthscale'] = rbf_l

    model = get_model(model_desc, active_learner, hyper, config)
    utility_f = get_fake_func(fake_function_desc, config)
    user_f = get_synth_user(synth_user, utility_f, config)

    eval_rew = [[env_p['rewards'] for env_p in data_env] for data_env in eval_data]
    train_rew = [[env_p['rewards'] for env_p in data_env] for data_env in train_data]
    eval_env_d = eval_data[env_num]
    
    # eval_user_d = np.empty((0, dim_rewards))
    # for path_d in eval_env_d:
    #     eval_user_d = np.append(eval_user_d, path_d['rewards'], axis=0)

    # M = 30

    # a_r = np.empty(M)
    # a_c = np.empty(M)
    # for j in range(M):

    #     user_f.fake_f.randomize()

    user_f.learn_beta(eval_rew, p_synth_pair, Q_size=2, p_sigma=p_synth_abs)

    # while True:
    #     try:
    #         print('Running tune beta')
    #         user_f.learn_beta(eval_user_d, p_synth_pair, Q_size=num_alts, p_sigma=p_synth_abs)
    #         print('Completed, breaking out of loop')
    #         break
    #     except:
    #         print('Unable to tune beta for user synth')
    #         # gonna randomize the function and try again
    #         utility_f.randomize()


        # ### Test user synth
        # train_env_d = train_data[env_num]
        # train_user_d = np.empty((0, dim_rewards))
        # for path_d in train_env_d:
        #     train_user_d = np.append(train_user_d, path_d['rewards'], axis=0)

        # N = 2500

        # count_choose = 0
        # count_rate = 0

        # for i in range(N):
        #     pair = np.random.choice(train_user_d.shape[0], 2, replace=False)

        #     idx = user_f.choose(train_user_d[pair])
        #     idx_rate = np.argmax(user_f.rate(train_user_d[pair]))
        #     corr_idx = np.argmax(user_f.fake_f(train_user_d[pair]))

        #     if corr_idx == idx:
        #         count_choose += 1

        #     if idx_rate == idx:
        #         count_rate += 1 

        # acc_rate = count_rate / N
        # acc = count_choose / N
        # print('Train Accuracy of choose is = ' + str(acc))
        # print('Train Accuracy of rate is = ' + str(acc_rate))

        # count_choose = 0
        # count_rate = 0

        # for i in range(N):
        #     pair = np.random.choice(eval_user_d.shape[0], 2, replace=False)

        #     idx = user_f.choose(eval_user_d[pair])
        #     idx_rate = np.argmax(user_f.rate(eval_user_d[pair]))
        #     corr_idx = np.argmax(user_f.fake_f(eval_user_d[pair]))

        #     if corr_idx == idx:
        #         count_choose += 1

        #     if idx_rate == idx:
        #         count_rate += 1 

        # acc_rate = count_rate / N
        # acc = count_choose / N

        # a_r[j] = acc_rate
        # a_c[j] = acc

        # print('Eval Accuracy of choose is = ' + str(acc))
        # print('Eval Accuracy of rate is = ' + str(acc_rate))

    # print('\nacc choice = '+str(a_c))
    # print('acc rate = '+str(a_r))

    # print('acc_choice mean=' + str(np.mean(a_c)) + ' std='+str(np.std(a_c)))
    # print('acc_rate mean=' + str(np.mean(a_r)) + ' std='+str(np.std(a_r)))

    # import matplotlib.pyplot as plt


    # plt.hist(a_c)
    # plt.title('Choice p_c='+str(p_synth_pair))
    # plt.figure()
    # plt.hist(a_r)
    # plt.title('Rating p_r='+str(p_synth_abs))

    # plt.show()

    # sys.exit(0)

    ###

    if config['add_model_prior']:
        model.add_prior(bounds = config['prior_bounds'], num_pts=config['prior_pts'])
    if config['add_abs_point'] and model_desc != 'linear':
        x = np.array([config['abs_point_loc']])
        y = np.array([config['abs_point_value']])
        model.add(x,y,type='abs')

    # setup storage of scores
    accuracy = np.zeros(num_training+1)
    avg_selection = np.zeros(num_training+1)
    query_type_is_abs = np.zeros(num_training+1, dtype=int) - 1
    query_is_correct = np.zeros(num_training+1, dtype=int) - 1
    all_ranks = np.zeros((num_training+1, len(eval_data[0])))

    estimated_scores = np.zeros((num_training+1, len(eval_data[0])))
    real_scores = np.zeros((num_training+1, len(eval_data[0])))
    score_diff = np.zeros((num_training+1, len(eval_data[0])))
    spearmans = np.zeros((num_training+1, len(eval_data[0])))
    pearsons = np.zeros((num_training+1, len(eval_data[0])))
    rhos = np.zeros((num_training+1, len(eval_data[0])))

    # test untrained model
    accuracy[0], avg_selection[0], all_ranks[0], estimated_scores[0], real_scores[0], score_diff[0], spearmans[0], pearsons[0], rhos[0] = \
                        evaluation(env_num, utility_f, config, model, eval_data)

    str_header = str(0)
    str_header.zfill(3)
    str_header = 'trainitr_'+str_header

    record_gp_state(model, utility_f, config['prior_bounds'], folder, \
                    file_header=str_header, visualize=True)

    bounds = [(0,0), (0,0)]
    rand_order = np.random.choice(len(train_data[env_num]), size=num_training, replace=False)

    ##### Start main training loop
    for itr, i in enumerate(rand_order):
        if verbose:
            print('\t\tNUMBER of X_train: ' + str(len(model.X_train)))


        ##### Extract rewards for environment domain
        rewards, indicies = train_data[env_num][i]['rewards'], train_data[env_num][i]['indicies']
        
        

        #### update bounds
        mins = np.amin(rewards, axis=1)
        maxs = np.amax(rewards, axis=1)
        for j in range(len(bounds)):
            if mins[j] < bounds[j][0]:
                bounds[j] = (mins[j], bounds[j][1])
            if maxs[j] > bounds[j][1]:
                bounds[j] = (bounds[j][0], maxs[j])

        if use_kmedoid == 'True' or use_kmedoid == 'true':
            if rewards.shape[0] > config['downselect_num']:
                kmed = KMedoids(n_clusters=60).fit(rewards)
                rewards = kmed.cluster_centers_
                print('Downselected using k-medoids')
        elif use_kmedoid == 'uni':
            if rewards.shape[0] > config['downselect_num']:
                subset = np.random.choice(np.arange(0, rewards.shape[0]), config['downselect_num'], replace=False)
                rewards = rewards[subset]
                print('Downselected using random')
        elif use_kmedoid == 'downhull':
            if rewards.shape[0] > config['downselect_num']:
                rewards = downsample_hull(rewards, config['downselect_num'])
                print('Downselected using downsample_hull')
        elif use_kmedoid == 'medrand':
            if rewards.shape[0] > config['downselect_num']:
                print('Starting downsampling process')
                rewards = downsample_kmed_rand(rewards, config['downselect_num'])
                print('Downselected using kmed_rand')

        if selection_type == 'choose1' or selection_type == 'ranking':
            if config['pareto_pairs']:
                sel_idx, non_shown_paths = model.select(rewards, num_alts, 
                                                        prefer_pts=len(indicies['pareto']),
                                                        return_not_selected=True)
            else:
                sel_idx = model.select(rewards, num_alts)
                non_shown_paths = []
            
            y_fake = utility_f(rewards[sel_idx])

            # get pairs
            if selection_type == 'choose1':
                y_pairs = user_f.choose_pairs(rewards[sel_idx])
            else:
                if synth_user != 'perfect':
                    raise(Exception('human choice not implemented for ranking'))
                y_pairs = lop.generate_ranking_pairs(rewards[sel_idx], utility_f)
            
            model.add(rewards[sel_idx], y_pairs)

            # add non shown pareto paths if desired
            if len(non_shown_paths) > 0:
                model.add(rewards[non_shown_paths], [])

            query_type_is_abs[itr+1] = PAIR_QUERY
            query_is_correct[itr+1] = -1
        elif selection_type == 'rating':
            sel_idx = model.select(rewards, 1)[0]

            #rating = rating_score_from_fake_f(rewards[sel_idx], utility_f, rating_bounds_global)
            rating = user_f.rate(rewards[sel_idx])
            rating_np = np.array([rating])

            model.add(rewards[np.newaxis,sel_idx], rating_np, type='abs')
            query_type_is_abs[itr+1] = ABS_QUERY
            query_is_correct[itr+1] = -1
        elif selection_type == 'switch':
            sel_idx = model.select(rewards, num_alts)
            x_train = rewards[sel_idx]
            

            # check if a rating or choose is selected by the active learning
            if len(sel_idx) == 1:
                rating = user_f.rate(x_train)

                model.add(x_train, rating, type='abs')
                query_type_is_abs[itr+1] = ABS_QUERY
            else:
                correct_idx = np.argmax(user_f.fake_f(x_train))
                y_pairs = user_f.choose_pairs(x_train)

                model.add(x_train, y_pairs)
                query_type_is_abs[itr+1] = PAIR_QUERY
                query_is_correct[itr+1] = (correct_idx == y_pairs[0][1])
                
            

        # end else if for selection type

        accuracy[itr+1], avg_selection[itr+1], all_ranks[itr+1], estimated_scores[itr+1], real_scores[itr+1], score_diff[itr+1], spearmans[itr+1], pearsons[itr+1], rhos[itr+1] = \
                        evaluation(env_num, utility_f, config, model, eval_data)

        str_header = str(itr+1)
        str_header.zfill(3)
        str_header = 'trainitr_'+str_header

        record_gp_state(model, utility_f, config['prior_bounds'], folder, \
                        file_header=str_header, visualize=True)
    # end for loop for training loop

    visualize_single_run_regret(folder, score_diff, query_type_is_abs)

    return accuracy, avg_selection, all_ranks, estimated_scores, real_scores, score_diff, query_type_is_abs, query_is_correct, spearmans, pearsons, rhos, path_data



def evaluation(env_num, utility_f, config, model, eval_data):
    num_correct = 0
    rank_sum = 0

    num_eval = len(eval_data[env_num])
    ranks = np.empty(num_eval)

    estimated_scores = np.empty(num_eval)
    real_scores = np.empty(num_eval)
    score_diff = np.empty(num_eval)
    spearmans = np.empty(num_eval)
    pearsons = np.empty(num_eval)
    rhos = np.empty(num_eval)
    score_diff = np.empty(num_eval)

    for i in range(num_eval):
        ##### Generate paths and select paths for explanation
        rewards, indicies = eval_data[env_num][i]['rewards'], eval_data[env_num][i]['indicies']
        
        rewards = rewards[indicies['pareto']]
        scores = model(rewards)

        best_idx = model.active_learner.select_best(scores, set(indicies['pareto']))
        indicies['best'] = best_idx

        fake_utility = utility_f(rewards)
        sorted_idx = np.argsort(fake_utility)[::-1]
        best_path = sorted_idx[0]
        selected_rank = np.where(sorted_idx == indicies['best'])[0][0]


        # calculate spearman and pearson correlation
        score_sort = np.argsort(scores)[::-1]

        spearman = np.corrcoef(np.array([sorted_idx, score_sort]))[0,1]
        pearson = np.corrcoef(np.array([fake_utility, scores]))[0,1]

        # calculate rho space
        rho_w = np.exp(scores)
        rho = np.exp(fake_utility)

        rho_w = rho_w / np.sum(rho_w)
        rho = rho / np.sum(rho)
        f_rho = -np.linalg.norm(rho - rho_w, ord=2)

        spearmans[i] = spearman
        pearsons[i] = pearson
        rhos[i] = f_rho
        estimated_scores[i] = scores[indicies['best']]
        real_scores[i] = fake_utility[indicies['best']]
        score_diff[i] = fake_utility[best_path] - fake_utility[indicies['best']]

        if selected_rank == 0:
            num_correct += 1

        rank_sum += selected_rank
        ranks[i] = selected_rank

    return num_correct / num_eval, rank_sum / num_eval, ranks, estimated_scores, real_scores, score_diff, spearmans, pearsons, rhos