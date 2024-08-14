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

# AbsAcquisition.py
# Written Ian Rankin - July 2024
#
# A learner function that uses an absloute query selection for use in combination
# with preference based learning algorithms.
#
#

import numpy as np
import math
import copy
from scipy.integrate import quad_vec
from scipy.stats import beta

from numba import jit

from lop.active_learning import AcquisitionBase
from lop.models import PreferenceGP, GP, PreferenceLinear

from lop.utilities import metropolis_hastings, sample_unique_sets

import pdb


#@jit(nopython=True)
def pq_integrand(q, aa, bb, f):
    # [w, Q]
    #p_q_w = numba_beta_pdf2(q, aa, bb)
    p_q_w = beta.pdf(q, aa, bb)

    M = f.shape[0]

    p_q = np.sum(p_q_w, axis=0) / M

    return fast_sum_Q(p_q_w, p_q, f, M)
    sum_Q = np.zeros(p_q_w.shape[1])

    for i in range(M):
        for j in range(M):
            sum_Q += f[i,j] * p_q_w[i,:] * p_q_w[j,:] / p_q

    return sum_Q / (M * M)

@jit(nopython=True)
def fast_sum_Q(p_q_w, p_q, f, M):
    sum_Q = np.zeros(p_q_w.shape[1])

    for i in range(M):
        for j in range(M):
            sum_Q += f[i,j] * p_q_w[i,:] * p_q_w[j,:] / p_q

    return sum_Q / (M * M)


class AbsAcquisition(AcquisitionBase):

    ## constructor
    # @param M - the number of samples to pull for calculating the expectation
    # @param rep_Q_method - the representative Q method to use
    # @param rep_Q_data - any data required for a particular representative Q method
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    def __init__(self, M=300, 
                 rep_Q_method = 'sampled', rep_Q_data = {'num_pts': 10, 'num_Q': 20},
                 alignment_f = 'rho',
                 default_to_pareto=False, always_select_best=False):
        super(AbsAcquisition, self).__init__(rep_Q_method=rep_Q_method,
                                                    rep_Q_data=rep_Q_data,
                                                    alignment_f=alignment_f,
                                                    default_to_pareto=default_to_pareto,
                                                    always_select_best=always_select_best)
        
        self.M = M


    




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
        size_query = len(prev_selection) + 1 # prev_selection + every new addition to Q
        N = len(mu)
        indicies = list(indicies)
        prev_selection = list(prev_selection)

        # get representative query for alignment function
        x_rep, Q_rep = self.get_representative_Q(candidate_pts)

        # If there is no data
        if x_rep is None:
            return np.random.choice(list(indicies), 1)[0]

        ## get sampled output from latent function
        all_rep, all_Q = self.get_samples_from_model(candidate_pts[indicies], x_rep)
        # all_Q [w, Q]

        ####### Calculate probability query for each sampled weight
        # Integral of p(q|w,Q)p(q|w',Q)
        # [w, w', Q]
        aa, bb = self.model.probits[2].get_alpha_beta(all_Q)


        ####### Calculate alignment function
        # [w, w']
        f = self.alignment(all_rep, Q_rep)


        ########## values post summation

        integ_p_q, err = quad_vec(pq_integrand, 0, 1, epsrel=0.001, 
                                    workers=-1, limit=200, args=(aa, bb, f))

        align_Q = integ_p_q

        best_idx = np.argmax(align_Q)
        self.sel_metric = align_Q[best_idx]
        return indicies[best_idx]


