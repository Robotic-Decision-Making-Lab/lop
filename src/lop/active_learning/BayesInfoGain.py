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

# BayesInfoGain.py
# Written Ian Rankin - April 2024
#
# A learner function that performs information gain on reward optimization.
#

import numpy as np
from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear

import pdb

class BayesInfoGain(ActiveLearner):

    # def __init__(self):
    #     self.sample_pts = None


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
        N = len(mu)
        indicies = list(indicies)

        if isinstance(self.model, PreferenceGP):
            K = self.model.cov




            # Get the probabilities of which candidate_pts is the best.
            p_B = np.ones(N) / N

            # precalculate the probit between each candidate_pts
            probit_mat = self.model.probits[0].likelihood_all_pairs(mu)
            log_probit = np.log(probit_mat)

            info_gain = np.zeros(len(indicies))

            # go through each possible query to calculate information gain on each
            for idx, Q_i in enumerate(indicies):
                
                Q = np.array(list(prev_selection) + [Q_i])

                if len(Q) < 2:
                    # THIS IS PROBABLY NOT THE RIGHT WAY TO HANDLE THIS
                    return np.random.choice(indicies)

                # IAN YOU ARE HERE, Consider how to properly handle B when it is q versus q_i
                B_idx = np.ones((N, len(Q)-1))
                B_idx[Q,np.arange(0,len(Q)-1,1)] = 0

                #p_q = 

                for i, q in enumerate(Q):
                    mask = np.ones(len(Q), bool)
                    mask[i] = False

                    probit_Q = probit_mat[q,Q[mask]]
                    p_q = np.prod(probit_Q)
                    log_p_q_info_gain = -np.log(probit_mat[q, Q[mask]]) + np.log(probit_mat[Q[mask], q])


                    info_gain_B = np.sum(p_B[:,np.newaxis] * B_idx @ log_p_q_info_gain[:,np.newaxis])

                    info_gain[idx] += info_gain_B * p_q
                    
            return indicies[np.argmax(info_gain)]

