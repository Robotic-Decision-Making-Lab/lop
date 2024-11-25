# load_intermediate_large.py
# Written Ian Rankin - November 2024

import lop

import numpy as np
import glob
from datetime import datetime as dt

import intermediate_loading as il


def main():
    parser = argparse.ArgumentParser(description='Load a model at an intermediate point')
    #parser.add_argument('--env', type=int, default=0, help='Environment number')
    #parser.add_argument('--trainitr', type=int, default=2, help='Train iteration')
    #parser.add_argument('--p_abs', type=str, default='0.7', help='p_abs')
    args = parser.parse_args()


    envs = [0,1,2,3,4,5,6,7,8,9]
    trainitrs = [2,4,6,8,10,12,14,16,18,20]
    #p_abs = args.p_abs
    p_abss = ['0.95', '0.9', '0.8', '0.7', '0.65']


    M=200
    #abs_comp = lop.AbsAcquisition(M=M, alignment_f='spearman')
    pair = lop.AcquisitionSelection(M=M, alignment_f='spearman', rep_Q_method='stable',
                            rep_Q_data={'filename': '../../../experiments/comparision_pts.npy'})
    abs_l = lop.UCBLearner(1.0)
    al = lop.AlignmentDecision(pairwise_l=pair, abs_l=abs_l, num_calls_decision=2,
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


    pref_count = np.zeros((len(p_abss), len(envs), len(trainitrs)))
    total_count = np.zeros((len(p_abss), len(envs), len(trainitrs)))
    rate_scores = np.zeros((len(p_abss), len(envs), len(trainitrs), 20))
    pref_scores = np.zeros((len(p_abss), len(envs), len(trainitrs), 20))

    for i_p, p_abs in enumerate(p_abss):
        for i_env, env in enumerate(envs):
            for i_itr, trainitr in enumerate(trainitrs):
                al.num_calls = trainitr
                al.calls_to_decision = trainitr
                
                for run in range(20):
                    model_path = il.get_filepath(trainitr=trainitr, run=run, env=env, 
                                            fake_f_name='min', 
                                            p_abs=p_abs, selector='SW_ALT_SPEAR')
                    il.model_load(model, model_path)

                    model.active_learner.num_calls = trainitr
                    sel_idxs, rating_score, pref_score = model.select(np.array([[1.0, 1.0, 1.0, 1.0, 1.0], [0,0,0,0,0]]), 2)
                    
                    print('rate_score = ' +str(rating_score) + ' pref_score = ' + str(pref_score))
                    rate_scores[i_p, i_env, i_itr, run] = rating_score
                    pref_scores[i_p, i_env, i_itr, run] = pref_score

                    print(sel_idxs)
                    if len(sel_idxs) > 1:
                        pref_count[i_p, i_env, i_itr] += 1
                    total_count[i_p,i_env, i_itr] += 1

                print('Env: ' + str(env) + ' itr: ' + str(trainitr))
                print(pref_count)
                print(total_count)
                print('\n\n\n\n\n')


    print(total_count)
    pref_pct = pref_count / total_count

    print('Total percentage selecting preference (p_abs, envs, trainitrs) = ')
    print(pref_pct)
    

    np.savez('result_p_abs_'+'stable'+'.npz', pref_pct=pref_pct, total_count=total_count, pref_count=pref_count, rate_scores=rate_scores, pref_scores=pref_scores)

    print('Pref pct. mean over (p_abs, trainitrs)')
    print(np.mean(pref_pct, axis=1))

    print('Pref pct. Mean over environments and train iterations')
    print(np.mean(pref_pct, axis=(1,2)))

if __name__ == '__main__':
    import argparse
    main()
