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

# UCBLearner.py
# Written Ian Rankin - December 2023
#
# Upper confidence bound learning algorithm

import numpy as np

from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear

class UCBLearner(ActiveLearner):
    ## Constructor
    # @param alpha - the scaler value on the UCB equation UCB = mean + alpha*sqrt(variance)
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, alpha=1, default_to_pareto=False, always_select_best=False):
        super(UCBLearner, self).__init__(default_to_pareto,always_select_best)
        self.alpha = alpha



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
        elif isinstance(self.model, PreferenceLinear):
            raise NotImplementedError("Have not implemented UCB with linear preferences")
        indicies = list(indicies)

        selected_UCB = mu[indicies] + self.alpha*np.sqrt(variance[indicies])

        return indicies[np.argmax(selected_UCB)]
