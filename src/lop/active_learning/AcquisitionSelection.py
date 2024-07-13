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

# AcquisitionSelection.py
# Written Ian Rankin - July 2024
#
# A learner function that uses a preference based acquisition selection for picking
# options to show to user.
# Based on:
# Evan Ellis, Gaurav R. Ghosal, Stuart J. Russell, Anca Dragan, Erdem Biyik, (2024)
# A Generalized Acquisition Function for Preference-based Reward Learning, in proc.
# International Conference on Robotics and Automation (ICRA)
#
#

import numpy as np
import math
from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear

from lop.utilities import metropolis_hastings, sample_unique_sets

import pdb

class AcquisitionSelection(ActiveLearner):

    ## constructor
    # @param M - the number of samples to pull for calculating the expectation
    # @param rep_Q_method - the representative Q method to use
    # @param rep_Q_data - any data required for a particular representative Q method
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, M=100, 
                 rep_Q_method = 'sampled', rep_Q_data = {'num_pts': 10, 'num_Q': 20},
                default_to_pareto=False, always_select_best=False):
        super(AcquisitionSelection, self).__init__(default_to_pareto,always_select_best)

        self.M = M
        self.rep_Q_method = rep_Q_method
        self.rep_Q_data = rep_Q_data



    ## get_representative_Q
    # This function gets a set of queries and target points.
    # Left as a function to allow a few different method to potentially be used.
    def get_representative_Q(self):
        # get the number of points in each query.
        try:
            num_alts = self.num_alts
        except:
            num_alts = 2
        
        if self.rep_Q_method == 'sampled':
            N = self.rep_Q_data['num_pts']
            num_Q = self.rep_Q_data['num_Q']

            # sample data points from previous model data
            num_X_train = len(self.model.X_train)
            N = min(num_X_train, N)
            X_pts = self.model.X_train[np.random.choice(num_X_train, N, replace=False)]

            num_Q = min(num_Q, math.comb(N, num_alts))

            Q = sample_unique_sets(N, num_Q, num_alts)


            return X_pts, Q
        else:
            raise ValueError("AcquisitionSelection get_representative_Q given an incorrect method type of: " + str(self.rep_Q_method))


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
        #raise NotImplementedError("AcquisitionSelection select_greedy is not implemented and has been called")
        N = len(mu)
        indicies = list(indicies)

        # get sampled possible output of latent functions
        if isinstance(self.model, (PreferenceGP, GP)):
            cov = self.model.cov

            # sample M possible parameters w (reward values of the GP)
            all_w = np.random.multivariate_normal(mu, cov, size=self.M)
        elif isinstance(self.model, PreferenceLinear):
            w_samples = metropolis_hastings(self.model.loss_func, self.M, dim=candidate_pts.shape[1])

            w_norm = np.linalg.norm(w_samples, axis=1)
            w_samples = w_samples / np.tile(w_norm, (2,1)).T
            # generate possible outputs from weighted samples
            all_w = (candidate_pts @ w_samples.T).T
        else:
            raise ValueError("Aquisition Selection select_greedy given an unknown model type + " + str(type(self.model)))
        

        # calculate the







    def select_pair(self, candidate_pts, mu, data, indicies, prev_selection, debug=True):
        raise NotImplementedError("AcquisitionSelection select_pair is not implemented and has been called")
