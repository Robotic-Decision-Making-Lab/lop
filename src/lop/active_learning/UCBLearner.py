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
    def __init__(self, alpha=1):
        super(UCBLearner, self).__init__()
        self.alpha = alpha

    ## select
    # Selects the given points
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @param prev_selection - [opt, default = []]a list of indicies that 
    # @param prefer_num - [default = None] the points at the start of the candidates
    #                   to prefer selecting from. Returned as:
    #                   a. A number of points at the start of canididate_pts to prefer
    #                   b. A set of points to prefer to select.
    #                   c. 'pareto' to indicate 
    #                   d. Enter 0 explicitly ignore selections
    #                   e. None (default) assumes 0 unless default to pareto is true.
    # @param return_not - [opt default-false] returns the not selected points when there
    #                   a preference to selecting to certian points. [] if not but set to true.
    #                   
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          only returns highest mean if "always select best is set"
    def select(self, candidate_pts, num_alts, prev_selection=[], prefer_pts=None, not_selected=False):
        prefer_pts = self.get_prefered_set_of_pts(candidate_pts, prefer_pts)

        if isinstance(self.model, (PreferenceGP, GP)):
            mu, variance = self.gp.predict(candidate_pts)
            UCB = mu + self.alpha*np.sqrt(variance)
        elif isinstance(self.model, PreferenceLinear):
            UCB = 1 # TODO
        else:
            raise Exception("UCBLearner does not know how to handle model of type: " + str(type(self.model)))

        best_idx = np.argmax(mu)

        selected_idx = self.select_best_k(UCB, num_alts, best_idx, prefer_num)

        return selected_idx, UCB[selected_idx], mu[best_idx]



    def select_greedy(self, cur_selection, data):
        mu, variance, cov, prefer_num = data

        best_v = -float('inf')
        best_i = -1

        exp_v = 1.0 / (len(cur_selection) + 1)
        for i in [x for x in range(len(mu)) if x not in cur_selection]:
            vari = variance[i]

            value = (1-self.alpha)*mu[i] + self.alpha*np.sqrt(vari)

            if value > best_v:
                best_v = value
                best_i = i

        return best_i, best_v
