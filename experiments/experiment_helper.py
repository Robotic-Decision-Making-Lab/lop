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
import pdb

import lop

from GP_visualization import record_gp_state, visualize_data

def get_active_learner(selector, selection_type, UCB_scalar, default_to_pareto, config, fake_func=None):
    #default_to_pareto = config['default_to_pareto']
    if selection_type == 'rating':
        always_select_best = False
    else:
        always_select_best = config['always_select_best']


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
    elif selector == 'BAYES_INFO_GAIN_999':
        al = lop.BayesInfoGain(default_to_pareto, always_select_best,p_q_B_method='999')
    elif selector == 'ACQ_RHO':
        al = lop.AcquisitionSelection(M=400, alignment_f='rho',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'ACQ_LL':
        al = lop.AcquisitionSelection(M=400, alignment_f='loglikelihood',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)   
    elif selector == 'ACQ_EPIC':
        al = lop.AcquisitionSelection(M=400, alignment_f='epic',
                                    default_to_pareto=default_to_pareto, 
                                    always_select_best=always_select_best)
    elif selector == 'ACQ_SPEAR':
        al = lop.AcquisitionSelection(M=400, alignment_f='spearman',
                                    default_to_pareto=default_to_pareto, 
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
                    cov_func=lop.RBF_kern(config['rbf_sigma'], config['rbf_lengthscale']),
                    normalize_gp=config['normalize_gp'],
                    pareto_pairs=pareto_pairs,
                    normalize_positive=config['normalize_postive'],
                    use_hyper_optimization=use_hyper,
                    active_learner= active_learner
        )
    elif model_desc == 'linear':
        model = lop.PreferenceLinear(pareto_pairs=pareto_pairs,\
                                    active_learner=active_learner)
    
    return model

def get_synth_user(user_desc, utility_f, config):
    if user_desc == 'perfect':
        func = lop.PerfectUser(utility_f)
    elif user_desc == 'human_choice':
        func = lop.HumanChoiceUser(utility_f)

    return func

def get_fake_func(fake_func_desc, config):
    if fake_func_desc == 'linear':
        func = lop.FakeLinear()
    elif fake_func_desc == 'squared':
        func = lop.FakeSquared()
    elif fake_func_desc == 'logistic':
        func = lop.FakeLogistic()
    elif fake_func_desc == 'sin_exp':
        func = lop.FakeSinExp()
    elif fake_func_desc == 'min':
        func = lop.FakeWeightedMin()
    elif fake_func_desc == 'max':
        func = lop.FakeWeightedMax() 
    elif fake_func_desc == 'squared_min_max':
        func = lop.FakeSquaredMinMax()

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


def train_and_eval(config_filename, 
                    env_num, 
                    folder,
                    selector='UCB',
                    selection_type='choose1',
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
                    verbose = False):
    #
    with open(config_filename, 'rb') as f:
        config = yaml.load(f.read(), Loader=yaml.Loader)

    with open('simple_rewards.rew', 'rb') as f:
        path_data = pickle.load(f)

    train_data = [path_d[:num_train] for path_d in path_data['train']]
    eval_data = [path_d[:num_eval] for path_d in path_data['eval']]

    if rbf_sigma is not None:
        config['rbf_sigma'] = rbf_sigma
    if UCB_scaler is None:
        UCB_scaler = config['UCB_scalar']

    num_training = num_train#10

    #### Get required models and fake functions
    active_learner = get_active_learner(selector, selection_type, UCB_scaler, default_pareto, config)
    model = get_model(model_desc, active_learner, hyper, config)
    utility_f = get_fake_func(fake_function_desc, config)
    user_f = get_synth_user(synth_user, utility_f, config)


    eval_user_d = np.empty((0,2))
    for eval_env_d in eval_data:
        for path_d in eval_env_d:
            eval_user_d = np.append(eval_user_d, path_d['rewards'], axis=0)

    try:
        user_f.learn_beta(eval_user_d, config['p_correct'], Q_size=num_alts)
    except:
        print('Unable to tune beta for user synth')

    if config['add_model_prior']:
        model.add_prior(bounds = config['prior_bounds'], num_pts=config['prior_pts'])
    if config['add_abs_point']:
        x = np.array([config['abs_point_loc']])
        y = np.array([config['abs_point_value']])
        model.add(x,y,type='abs')

    # setup storage of scores
    accuracy = np.zeros(num_training+1)
    avg_selection = np.zeros(num_training+1)
    all_ranks = np.zeros((num_training+1, len(eval_data[0])))

    estimated_scores = np.zeros((num_training+1, len(eval_data[0])))
    real_scores = np.zeros((num_training+1, len(eval_data[0])))
    score_diff = np.zeros((num_training+1, len(eval_data[0])))

    # test untrained model
    accuracy[0], avg_selection[0], all_ranks[0], estimated_scores[0], real_scores[0], score_diff[0] = \
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

        elif selection_type == 'rating':
            sel_idx = model.select(rewards, 1)[0]

            #rating = rating_score_from_fake_f(rewards[sel_idx], utility_f, rating_bounds_global)
            rating = user_f.rate(rewards[sel_idx])
            rating_np = np.array([rating])

            model.add(rewards[np.newaxis,sel_idx], rating_np, type='abs')
        # end else if for selection type

        accuracy[itr+1], avg_selection[itr+1], all_ranks[itr+1], estimated_scores[itr+1], real_scores[itr+1], score_diff[itr+1] = \
                        evaluation(env_num, utility_f, config, model, eval_data)

        str_header = str(itr+1)
        str_header.zfill(3)
        str_header = 'trainitr_'+str_header

        record_gp_state(model, utility_f, config['prior_bounds'], folder, \
                        file_header=str_header, visualize=True)
    # end for loop for training loop

    return accuracy, avg_selection, all_ranks, estimated_scores, real_scores, score_diff



def evaluation(env_num, utility_f, config, model, eval_data):
    num_correct = 0
    rank_sum = 0

    num_eval = len(eval_data[env_num])
    ranks = np.empty(num_eval)

    estimated_scores = np.empty(num_eval)
    real_scores = np.empty(num_eval)
    score_diff = np.empty(num_eval)

    for i in range(num_eval):
        ##### Generate paths and select paths for explanation
        rewards, indicies = eval_data[env_num][i]['rewards'], eval_data[env_num][i]['indicies']
        scores = model(rewards)

        best_idx = model.active_learner.select_best(scores, set(indicies['pareto']))
        indicies['best'] = best_idx

        fake_utility = utility_f(rewards)
        sorted_idx = np.argsort(fake_utility)[::-1]
        best_path = sorted_idx[0]
        selected_rank = np.where(sorted_idx == indicies['best'])[0][0]

        estimated_scores[i] = scores[indicies['best']]
        real_scores[i] = fake_utility[indicies['best']]
        score_diff[i] = fake_utility[best_path] - fake_utility[indicies['best']]

        if selected_rank == 0:
            num_correct += 1

        rank_sum += selected_rank
        ranks[i] = selected_rank

    return num_correct / num_eval, rank_sum / num_eval, ranks, estimated_scores, real_scores, score_diff