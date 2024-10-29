# HumanChoiceUser2.py
# Written Ian Rankin - October 2024
#
# Implementation of synthetic users that sets up the beta and gamma code differently than
# previous methods.

import numpy as np
import pdb

from copy import copy

from lop.utilities import HumanChoiceUser
from lop.probits import std_norm_cdf
from scipy.optimize import minimize_scalar
import scipy.stats as st
import scipy.special as spec
import itertools

class HumanChoiceUser2(HumanChoiceUser):

    ## constructor
    ## constructor
    # @param fake_f - the fake function for the user.
    def __init__(self, fake_f, beta = 1.0, sigma = 1.0, bot_range=0.1, top_range=0.9):
        super(HumanChoiceUser2, self).__init__(fake_f, beta, sigma, bot_range, top_range)


    def obj(self, b, diffs, p):
        eb = np.exp(np.abs(diffs) * b)
        
        logs = eb / (1+eb)
        logs = np.where(np.isnan(logs), 1, logs)
        logs = np.where(logs < 0.50000001, np.nan, logs)

        obj_sum = np.nanmean(logs)

        return (obj_sum - p)**2
    
    def rate_obj(self, sigma, diffs, p):
        diffs = np.abs(diffs)
        sig_scalar = 1.0 / (np.sqrt(2) * sigma)


        F = spec.ndtr(diffs * sig_scalar)

        F = np.where(diffs == 0.0, np.nan, F)
        avg = np.nanmean(F)

        return (avg - p)**2





    def get_k_b(self, f):
        min_sigmoid = np.log(self.bot_range) - np.log(1 - self.bot_range)
        max_sigmoid = np.log(self.top_range) - np.log(1 - self.top_range)

        f_max = np.max(f)
        f_min = np.min(f)
        k = (max_sigmoid - min_sigmoid) / (f_max - f_min)
        b = min_sigmoid - k * f_min

        return k, b

    ## learn_beta
    # This function learns the required beta value and sigma for both 
    # pairwise and absloute queries
    # @param rewards - a set of reward values to try
    # @param p - the probability of selecting the best value
    # @param num_Q - [opt default=2] the number of points in each query
    def learn_beta(self, rewards, p, Q_size=2, p_sigma=None):
        if p_sigma is None:
            p_sigma = p
        min_path_problems = min([min([len(problem) for problem in env]) for env in rewards])
        
        min_path_problems = min([min_path_problems, 500])

        num_functions = 1
        fake_fs = [self.fake_f]
   



        # collapase to a single sampled matrix
        F = np.empty((num_functions, len(rewards) * len(rewards[0]), min_path_problems))
        for f_idx in range(num_functions):
            f_list = [[fake_fs[f_idx](r_problem) for r_problem in r_env] for r_env in rewards]
            min_max = np.array([min([min([np.min(f1) for f1 in f2]) for f2 in f_list]),  \
                                max([max([np.max(f1) for f1 in f2]) for f2 in f_list])])
            print(min_max)
            k,b = self.get_k_b(min_max)

            # through environments
            for i in range(len(rewards)):
                # through planning problems
                for j in range(len(rewards[i])):
                    idx = i * len(rewards[0]) + j

                    samp_idx = np.random.choice(len(rewards[i][j]), min_path_problems, replace=False)
                    f = fake_fs[f_idx](rewards[i][j])
                    f = f * k + b

                    F[f_idx, idx] = f[samp_idx]

        pairs = np.array(list(itertools.combinations(range(F.shape[2]), 2)), dtype=int)
        diffs = F[:,:, pairs[:,0]] - F[:,:,pairs[:,1]]


        ## rating tuning
        #print('p_sigma = ' + str(p_sigma))
        res = minimize_scalar(self.rate_obj, bounds=[0.001, 5.0], args=(diffs, p_sigma), options={'xatol': 0.001})

        self.sigma = res.x
        #print('Rating tuning')
        #print(res)

        ## Pairwise tuning
        res = minimize_scalar(self.obj, bounds=[0.01, 200.0], args=(diffs, p), options={'xatol': 0.001})

        f_list = [[self.fake_f(r_problem) for r_problem in r_env] for r_env in rewards]
        min_max = np.array([min([min([np.min(f1) for f1 in f2]) for f2 in f_list]),  \
                            max([max([np.max(f1) for f1 in f2]) for f2 in f_list])])            

        self.k, self.b = self.get_k_b(min_max)
        self.beta = res.x

        




