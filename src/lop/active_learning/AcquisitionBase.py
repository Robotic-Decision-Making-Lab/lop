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

# AcquisitionBase.py
# Written Ian Rankin - July 2024
#
# A base set of code for acquisition selection.
# Not implementable by itself, just a base for different versions of acquisition function
# Alignment functions based on below:
# Evan Ellis, Gaurav R. Ghosal, Stuart J. Russell, Anca Dragan, Erdem Biyik, (2024)
# A Generalized Acquisition Function for Preference-based Reward Learning, in proc.
# International Conference on Robotics and Automation (ICRA)
#
#

import numpy as np
import math
import copy
from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear

from lop.utilities import metropolis_hastings, sample_unique_sets

import pdb

class AcquisitionBase(ActiveLearner):

    ## constructor
    # @param rep_Q_method - the representative Q method to use
    # @param rep_Q_data - any data required for a particular representative Q method
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, 
                rep_Q_method = 'sampled', rep_Q_data = {'num_pts': 10, 'num_Q': 20},
                alignment_f = 'rho',
                default_to_pareto=False, always_select_best=False):
        super(AcquisitionBase, self).__init__(default_to_pareto,always_select_best)

        self.rep_Q_method = rep_Q_method
        self.rep_Q_data = rep_Q_data
        self.alignment_f = alignment_f

        self.samples_set = False



    ## get_representative_Q
    # This function gets a set of queries and target points.
    # Left as a function to allow a few different method to potentially be used.
    def get_representative_Q(self, candidate_pts=None):
        # get the number of points in each query.
        try:
            num_alts = self.num_alts
            if num_alts == 1:
                num_alts = 2
        except:
            num_alts = 2
        
        if self.rep_Q_method == 'sampled':
            N = self.rep_Q_data['num_pts']
            num_Q = self.rep_Q_data['num_Q']

            # sample data points from previous model data
            if self.model.X_train is None:
                return None, None
            X_train = self.model.X_train
            num_X_train = len(self.model.X_train)
            
            if num_X_train < 4:
                X_train = np.append(X_train, candidate_pts, axis=0)
                num_X_train = X_train.shape[0]

            N = min(num_X_train, N)
            X_pts = X_train[np.random.choice(num_X_train, N, replace=False)]

            num_Q = min(num_Q, math.comb(N, num_alts))

            Q = sample_unique_sets(N, num_Q, num_alts)


            return X_pts, Q
        else:
            raise ValueError("AcquisitionSelection get_representative_Q given an incorrect method type of: " + str(self.rep_Q_method))



    ## alignment
    # this allignment function is the f(R_w, R_w') defined in the paper
    # This is a score of how similar the two reward functions are to each other.
    # @param all_rep - scores of each sampled weight function with represenative points
    #                   (M_samples num_rep) with sampled score
    # @param Q_rep - the set of represantive queries
    def alignment(self, all_rep, Q_rep):
        if all_rep.shape[1] < 2:
            return np.zeros((all_rep.shape[0], all_rep.shape[0]))
        
        if self.alignment_f == 'rho':
            rho_R = np.exp(all_rep)
            rho_R_sum = np.sum(rho_R, axis=1)
            rho_R = rho_R / np.repeat(rho_R_sum[:,np.newaxis], all_rep.shape[1], axis=1)

            diff = np.repeat(rho_R[np.newaxis,:,:], rho_R.shape[0], axis=0) - np.repeat(rho_R[:,np.newaxis,:], rho_R.shape[0], axis=1)
            
            f_rho = -np.linalg.norm(diff, axis=2, ord=2)

            return f_rho
        elif self.alignment_f == 'loglikelihood':
            if all_rep.shape[1] < 2:
                return np.ones((all_rep.shape[0], all_rep.shape[0]))

            probit_mat = np.array([self.model.probits[0].likelihood_all_pairs(w) for w in all_rep])
            


            q_best_w = np.argmax(all_rep[:,Q_rep], axis=2)

            p_q = np.zeros((self.M, Q_rep.shape[1], Q_rep.shape[0]))

             # this could be vectorized to be faster
            for i in range(Q_rep.shape[1]):
                p_q_i = np.ones((self.M, Q_rep.shape[0]))
                for j in range(Q_rep.shape[1]):
                    # calculate p_q for each q in Q
                    if i != j:
                        p_q_i *= probit_mat[:,Q_rep[:,i], Q_rep[:,j]]
                p_q[:, i, :] = p_q_i#np.prod(probit_mat[:,Q_rep[:,i], Q_rep], axis=2) * 2.0

            # normalize p_q to ensure it is correct
            sum_p_q = np.repeat(np.sum(p_q, axis=1)[:,np.newaxis,:], p_q.shape[1], axis=1)
            
            # p_q_Q [w, q, Q]
            p_q = p_q / sum_p_q

            # calculate the probability of p(q=argmax Rw| Q,Rw')
            p_q_mod = np.swapaxes(p_q, 1,2)
            g = np.sum(np.log(p_q_mod[:,np.arange(Q_rep.shape[0]), q_best_w]), axis=2)

            f = g + g.T
            return f
            
        elif self.alignment_f == 'epic':
            # This is a naive approach that assumes the reward is already invariant to
            # potential shaping
            # This makes sense, since reward moving earlier or later in time does not
            # really make sense in the full trajectory metric.
            # However, the pearson correlation between represantive samples still makes sense
            # for cases without particular actions

            
            pear_dis = np.sqrt(1 - np.corrcoef(all_rep)) / np.sqrt(2)
            return -pear_dis
        elif self.alignment_f == 'spearman':
            ranked = np.argsort(all_rep, axis=1)

            spearman = np.corrcoef(ranked)
            spear_dis = np.sqrt(1 - spearman) / np.sqrt(2)
            return -spear_dis
            return spearman

        elif self.alignment_f == 'one':
            # will not work for anything meaningful, but can be used to check pipeline of working code
            return np.ones((all_rep.shape[0], all_rep.shape[0]))

    def set_samples(self, all_rep, all_Q):
        self.all_rep = all_rep
        self.all_Q = all_Q
        self.samples_set = True

    def unset_samples(self):
        self.samples_set = False

    def get_samples_from_model(self, candidate_pts, x_rep, indicies=None):
        if self.samples_set:
            if indicies is not None:
                return self.all_rep, self.all_Q[:,indicies]
            return self.all_rep, self.all_Q
        
        N = candidate_pts.shape[0]

        ## get sampled possible output of latent functions
        if isinstance(self.model, (PreferenceGP, GP)):
            cov = self.model.cov

            x_both = np.append(candidate_pts, x_rep, axis=0)

            # need to sample both representive and query samples at the same time.
            mu_both, simga_both = self.model.predict(x_both)
            cov_both = self.model.cov

            # sample M possible parameters w (reward values of the GP)
            all_samples = np.random.multivariate_normal(mu_both, cov_both, size=self.M)
            all_Q = all_samples[:, :N]
            all_rep = all_samples[:, N:]
        elif isinstance(self.model, PreferenceLinear):
            w_samples = metropolis_hastings(self.model.loss_func, self.M, dim=candidate_pts.shape[1])

            #w_norm = np.linalg.norm(w_samples, axis=1)
            #w_samples = w_samples / np.tile(w_norm, (candidate_pts.shape[1],1)).T
            # generate possible outputs from weighted samples
            all_rep = (x_rep @ w_samples.T).T
            all_Q = (candidate_pts @ w_samples.T).T
        else:
            raise ValueError("Aquisition Selection get_samples_from_model given an unknown model type + " + str(type(self.model)))

        return all_rep, all_Q

    ## select_greedy
    # This function greedily selects the best single data point
    # Depending on the selection method, you are not forced to implement this function
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param mu - a numpy array of mu values outputed from predict. numpy (n)
    # @param data - a user defined tuple of data (determined by the predict function of the model)
    # @param indicies - a list or set of indicies in candidate points to consider.
    # @param prev_selection - a set ofindicies of previously selected points
    #
    # @return the index of the greedy selection.
    def select_greedy(self, candidate_pts, mu, data, indicies, prev_selection):
        raise NotImplementedError("select_greedy AcquisitonBase not implemented")
