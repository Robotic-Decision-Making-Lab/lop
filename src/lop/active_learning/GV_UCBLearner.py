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

# GV_UCBLearner.py
# Written Ian Rankin - December 2023
#
# A learner function that uses the generalized variance instead of just variance
# to select the next value.

import numpy as np

from lop.active_learning import UCBLearner
from lop.models import PreferenceGP, GP, PreferenceLinear
from lop.utilities import metropolis_hastings

import pdb

class GV_UCBLearner(UCBLearner):
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
        if isinstance(self.model, (PreferenceGP, GP)):
            variance = data
            cov = self.model.cov
        elif isinstance(self.model, PreferenceLinear):
            w_samples = metropolis_hastings(self.model.loss_func, 200, dim=candidate_pts.shape[1])

            #w_norm = np.linalg.norm(w_samples, axis=1)
            #w_samples = w_samples / np.tile(w_norm, (2,1)).T
            # generate possible outputs from weighted samples
            all_w = (candidate_pts @ w_samples.T).T

            cov = np.cov(all_w.T)
            variance = np.diagonal(cov)
        indicies = list(indicies)
        prev_selection = list(prev_selection)

        exp_v = 1.0 / (len(prev_selection)+1)

        # calculate the Standardized general variance
        # The matrix covariance equivelant to the standard deviation
        SGV = [np.linalg.det(cov[np.ix_(prev_selection+[idx], prev_selection+[idx])]) ** exp_v for idx in indicies]

        selected_GV_UCB = mu[indicies] + self.alpha*np.array(SGV)

        best_idx = np.argmax(selected_GV_UCB)
        self.sel_metric = selected_GV_UCB[best_idx]
        return indicies[best_idx]
        


