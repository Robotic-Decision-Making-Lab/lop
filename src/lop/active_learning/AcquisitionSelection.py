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
import copy
from lop.active_learning import AcquisitionBase
from lop.models import PreferenceGP, GP, PreferenceLinear

from lop.utilities import metropolis_hastings, sample_unique_sets

from numba import jit

import pdb




@jit(nopython=True)
def quick_calculation_sum_align_q(f, p_q):
    sum_align_q = np.zeros((p_q.shape[1], p_q.shape[1]))

    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            sum_align_q += f[i,j] * p_q[i] * p_q[j]

    return sum_align_q



class AcquisitionSelection(AcquisitionBase):

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
        super(AcquisitionSelection, self).__init__(rep_Q_method=rep_Q_method,
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

        if size_query == 1:
            # THIS IS PROBABLY NOT THE RIGHT WAY TO HANDLE THIS
            # But needs at least a pair in order to calculate properly.
            self.sel_metric = 0
            return np.random.choice(indicies)[0]
            #return np.argmax(mu)

        x_rep, Q_rep = self.get_representative_Q(candidate_pts)

        if x_rep is None:
            self.sel_metric = 0
            return np.random.choice(list(indicies), 1)[0]

        ## get sampled output from latent function
        all_rep, all_Q = self.get_samples_from_model(candidate_pts, x_rep)
        
        
        # precalculate the probit between each candidate_pts
        probit_mat_Q = np.array([self.model.probits[0].likelihood_all_pairs(w) for w in all_Q])
        #probit_mat_rep = np.array([self.model.probits[0].likelihood_all_pairs(w) for w in all_rep])




        ###### calculate the p_q for each Q {indicies} + prev_selection

        # p_q_given_Q_w (M_samples, q, Q_new)
        p_q = np.ones((self.M, size_query, len(indicies)))
        
        # calculate the p_q across the previous selection
        if len(prev_selection) > 1:
            for i in range(0, len(prev_selection)):
                q_idx = prev_selection[i]
                prev_except = copy.copy(prev_selection)
                del prev_except[i]

                p_q_i = np.prod(probit_mat_Q[:,q_idx, prev_except],axis=1)
                p_q[:,i,:] = np.repeat(p_q_i[:,np.newaxis], len(indicies), axis=1)

        # calculate p_q across the current possible selections
        p_q[:,-1, :] = np.prod(probit_mat_Q[:, indicies,:][:,:,prev_selection], axis=2)

        # and back calculate across possible selections the previous selections
        p_q[:,:-1, :] *= probit_mat_Q[:, prev_selection,:][:,:,indicies]
        
        # normalize p_q given sum over q
        sum_p_q = np.repeat(np.sum(p_q, axis=1)[:,np.newaxis,:], p_q.shape[1], axis=1)
        p_q = p_q / sum_p_q

        

        ###### calculate alignment function
        f = self.alignment(all_rep, Q_rep)
        # [w,w', q, Q_new]
        f_expand = np.repeat(np.repeat(f[:,:,np.newaxis], p_q.shape[1],axis=2)[:,:,:,np.newaxis], p_q.shape[2], axis=3)

        ##### calculate expected alignment
        # equation (10)
        p_q_w0 = np.repeat(p_q[np.newaxis, :,:,:], self.M, axis=0)
        p_q_w1 = np.repeat(p_q[:, np.newaxis,:,:], self.M, axis=1)
        align_expand = p_q_w0 * p_q_w1 * f_expand
        E_align_q = np.sum(align_expand, axis=(0,1)) / (self.M * self.M)
        E_p_q = np.mean(p_q, axis=0)

        align_Q = E_align_q / E_p_q

        align_Q = np.sum(align_Q, axis=0)


        best_idx = np.argmax(align_Q)
        self.sel_metric = align_Q[best_idx]

        return indicies[best_idx]




    def select_pair(self, candidate_pts, mu, data, indicies, prev_selection, debug=True):
        x_rep, Q_rep = self.get_representative_Q(candidate_pts)
        if x_rep is None:
            idxs = np.random.choice(list(indicies), 2, replace=False)
            self.sel_metric = 0
            return (idxs[0], idxs[1])

        ## get sampled output from latent function
        all_rep, all_Q = self.get_samples_from_model(candidate_pts, x_rep)
        

        # precalculate the probit between each candidate_pts
        probit_mat_Q = np.array([self.model.probits[0].likelihood_all_pairs(w) for w in all_Q])

        # nothing needs to happen the p_q is already the probit mat since it is pairwise
        # p_q[k,i,j] = p(q=i | Q=(i,j)) for sample k or rather p_q[k, 0,1] = p(F_k(0) > F_k(1))
        # [w, Q, Q]
        p_q = probit_mat_Q


        ##### Calculate alignment function
        # [w, w']
        f = self.alignment(all_rep, Q_rep)
        
        

        ##### calculate expected alignment
        # equation (10)

        if self.M * self.M * p_q.shape[1] * p_q.shape[1] > 1000:
            sum_align_q = quick_calculation_sum_align_q(f, p_q)

            E_align_q = sum_align_q / (self.M * self.M)
        else:

            # [w,w', Q, Q]
            f_expand = np.repeat(np.repeat(f[:,:,np.newaxis], p_q.shape[1],axis=2)[:,:,:,np.newaxis], p_q.shape[2], axis=3)


            # [w, w', Q, Q]
            p_q_w0 = np.repeat(p_q[np.newaxis, :,:,:], self.M, axis=0)
            # [w', w, Q,Q]
            p_q_w1 = np.repeat(p_q[:, np.newaxis,:,:], self.M, axis=1)

            align_expand = p_q_w0 * p_q_w1 * f_expand
            # [Q, Q]
            E_align_q = np.sum(align_expand, axis=(0,1)) / (self.M * self.M)
        
        E_p_q = np.mean(p_q, axis=0)

        E_align_q = E_align_q / E_p_q

        ################## Add a print statement here, is E_p_q a weird number???
        #pdb.set_trace()

        # represents the alignment metric for each pair being selected as align_Q
        align_Q = E_align_q + E_align_q.T
        
        pair = self.pick_pair_from_metric(align_Q, prev_selection)
        self.sel_metric = align_Q[pair[0], pair[1]]
        return pair
