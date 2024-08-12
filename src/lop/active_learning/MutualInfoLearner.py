# Copyright 2023 Ian Rankin
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

# MutualInfLearner.py
# Written Ian Rankin - December 2023
#
# A set of code to select active learning algorithms for user preferences.
# This code implements the mutual information model described in 
# [1] Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh
#
#  This method only needs
# p(q|w,Q) human choice model given w = rewards and Q is the particular query.
# by sampling from the distribution of potential weights. 

import numpy as np

from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear

from lop.utilities import p_human_choice, metropolis_hastings

class MutualInfoLearner(ActiveLearner):
    ## Constructor
    # @param fake_fun - [opt default None] A fake function to scale the parameter space for GPs? (I actually don't remember why this exists)
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, fake_func=None, default_to_pareto=False, always_select_best=False):
        super(MutualInfoLearner, self).__init__(default_to_pareto,always_select_best)
        self.M = 75 # random value at the moment
        self.peakiness = 10
        self.fake_func = fake_func



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
        indicies = list(indicies)
        prev_selection = list(prev_selection)
        if isinstance(self.model, (PreferenceGP, GP)):
            variance = data
            cov = self.model.cov

            # sample M possible parameters w (reward values of the GP)
            all_w = np.random.multivariate_normal(mu, cov, size=self.M)
        elif isinstance(self.model, PreferenceLinear):
            w_samples = metropolis_hastings(self.model.loss_func, self.M, dim=candidate_pts.shape[1])

            w_norm = np.linalg.norm(w_samples, axis=1)
            w_samples = w_samples / np.tile(w_norm, (2,1)).T
            # generate possible outputs from weighted samples
            all_w = (candidate_pts @ w_samples.T).T
            
        if self.fake_func is not None:
            fake_f_mean = np.mean(self.fake_func(candidate_pts))
            samp_mean = np.mean(all_w)

            print('Scaling using fake function: ' + str(fake_f_mean / samp_mean))
            all_w = all_w * (fake_f_mean / samp_mean)


        info_gain = [self.calc_info_gain(prev_selection + [idx], all_w) for idx in indicies]

        best_idx = np.argmax(info_gain)
        self.sel_metric = info_gain[best_idx]
        return indicies[best_idx]
        

    # calculate the info gain for a query Q given the sampled parameters / reward W
    # only need p(q|w,Q) human choice model given w = rewards and Q is the particular query.
    # shouldn't this need p(w) as well? No because it is sampled from the distribution
    # Can I solve that exactly with the GP?
    #
    # @param Q - list of indicies of query.
    # @param all_w - a matrix of possible rewards for sample set of parameters [M,N]
    #                   M - number of samples
    #                   N - dimension of candidate points.
    #
    def calc_info_gain(self, Q, all_w):
        # Find the probabilities of human selecting a query given the possible reward values
        p = p_human_choice(all_w[:,Q], self.peakiness)
        # find the sum of the probabilities of w
        sum_p_over_w = np.sum(p, axis=0)

        # Find the information gain using the sample equation (4) in [1]
        info_gain = np.sum(p * np.log2(self.M * p / sum_p_over_w)) / self.M

        return info_gain
        

