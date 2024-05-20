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
from lop.utilities import calc_cdf, metropolis_hastings


from copy import deepcopy

import pdb









class BayesInfoGain(ActiveLearner):

    # def __init__(self):
    #     self.sample_pts = None


    

    ## p_B_pref_gp
    # Calculates the probability of each pt in the given matrix as being the being the best path
    # but only does it for preference GPs
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param mu - a numpy array of mu values outputed from predict. numpy (n)
    def p_B_pref_gp(self, candidate_pts, mu, cdf_method='auto'):
        K = self.model.cov

        p = np.empty(len(candidate_pts))

        # calculate the combined covariance matrix for if point i is the largest point.
        for i in range(len(candidate_pts)):
            K_star_i = np.zeros((len(p)-1, len(p)-1))

            idx_i = list(range(len(p)))
            idx_i.remove(i)

            

            for j in range(len(K_star_i)):
                for k in range(len(K_star_i)):
                    K_star_i[j,k] = K[i, i] + K[idx_i[j], idx_i[k]] - K[i, idx_i[j]] - K[i, idx_i[k]]

            
            sig = self.model.probits[0].sigma
            #K_star_i += np.diag(np.ones(len(idx_i)) * 2 * sig * sig)
            #K_star_i += np.ones((len(K_star_i), len(K_star_i))) * 2 * sig

            mu_star = mu[idx_i] - mu[i]
            #mu_star = mu[i] - mu[idx_i]


            p[i] = calc_cdf(mu_star, K_star_i, method=cdf_method)

        #print('p_sum = ' + str(np.sum(p)))
        p = p / np.sum(p)
        return p

    ## p_B_pref_gp
    # Calculates the probability of each pt in the given matrix as being the being the best path
    # but only does it for preference GPs
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param mu - a numpy array of mu values outputed from predict. numpy (n)
    def p_B_pref_linear(self, candidate_pts, mu, probit_mat):
        p = np.zeros(len(candidate_pts))

        #p = np.sum(np.log(probit_mat), axis=1) - np.log(probit_mat[0,0]) # * 2 multiplies the diagonal element (always 0.5)
        #p = np.exp(p)

        # sampling weights from linear model
        w_samples = metropolis_hastings(self.model.loss_func, 2000, dim=candidate_pts.shape[1])

        w_norm = np.linalg.norm(w_samples, axis=1)
        w_samples = w_samples / np.tile(w_norm, (2,1)).T
        # generate possible outputs from weighted samples
        all_w = (candidate_pts @ w_samples.T).T

        # frequentist approach from bayesian samples (not sure that's the correct term)
        largest_sample = np.argmax(all_w, axis=1)
        for s in largest_sample:
            p[s] += 1

        #print('p_sum = ' + str(np.sum(p)))
        p = p / np.sum(p)
        return p

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



            # Get the probabilities of which candidate_pts is the best.
            p_B = self.p_B_pref_gp(candidate_pts, mu)

            # precalculate the probit between each candidate_pts
            probit_mat = self.model.probits[0].likelihood_all_pairs(mu)
            print(probit_mat)
            log_probit = np.log2(probit_mat)

            info_gain = np.zeros(len(indicies))
            info_gain2 = np.zeros(len(indicies))

            # go through each possible query to calculate information gain on each
            for idx, Q_i in enumerate(indicies):
                
                Q = np.array(list(prev_selection) + [Q_i])

                if len(Q) < 2:
                    # THIS IS PROBABLY NOT THE RIGHT WAY TO HANDLE THIS
                    #return np.random.choice(indicies)
                    return np.argmax(p_B)

                p_q = np.zeros(len(Q))
                for i, q in enumerate(Q):
                    mask = np.ones(len(Q), bool)
                    mask[i] = False

                    probit_Q = probit_mat[q,Q[mask]]
                    p_q[i] = np.prod(probit_Q)
                
                p_q = p_q / np.sum(p_q)

                p_q_B = np.zeros((len(Q), len(Q)))
                for i, q in enumerate(Q):
                    #Q_new = deepcopy(Q)
                    Q_new = np.delete(Q, i)

                    for j, b in enumerate(Q):
                        p_q_b = 1
                        for k, q_i in enumerate(Q_new):
                            if q_i == b:
                                p_q_b *= 0.01#max(probit_mat[q_i, q], probit_mat[q, q_i])
                            else:
                                p_q_b *= 0.99#min(probit_mat[q,q_i], probit_mat[q_i, q])
                        p_q_B[i,j] = p_q_b                        

                #pdb.set_trace()
                p_q_B = p_q_B / np.sum(p_q_B, axis=0)
                print('p_q_B: ' + str(p_q_B))

                tmp = (-np.repeat(np.log2(p_q[:,np.newaxis]),2,axis=1) + np.log2(p_q_B))

                info_gain2[idx] = -np.sum(p_q * p_B[Q] * tmp)

                for i, q in enumerate(Q):
                    mask = np.ones(len(Q), bool)
                    mask[i] = False

                    # probit_Q = probit_mat[q,Q[mask]]
                    # p_q = np.prod(probit_Q)


                    #for j, q_i in enumerate(Q[mask]):
                    #    info_gain_B += p_B[q_i] * p_q * (log_probit[q, q_i] + log_probit[q_i, q])

                    info_gain_B = np.sum(p_B[Q[mask]] * (-log_probit[q, Q[mask]] + log_probit[Q[mask], q]))
                    #info_gain_B = np.sum(p_B[q] * (-log_probit[q, Q[mask]] + log_probit[Q[mask], q]))


                    # B_idx = np.zeros((N, len(Q)-1))
                    # # 1 where q_i == W_i
                    # pdb.set_trace()
                    # B_idx[q, :] = 0

                    # probit_Q = probit_mat[q,Q[mask]]
                    # p_q = np.prod(probit_Q)
                    # log_p_q_info_gain = -np.log2(probit_mat[q, Q[mask]]) + np.log2(probit_mat[Q[mask], q])

                    # info_gain_B = np.sum(p_B[:,np.newaxis] * B_idx @ log_p_q_info_gain[:,np.newaxis])
                    print('q: ' + str(q) + ' i: ' + str(i) + ' info_gain_B: ' + str((-log_probit[q, Q[mask]] + log_probit[Q[mask], q])) + ' p(q): ' + str(p_q[i]) + ' Q: ' + str(Q) + ' p_B[q_mask] ' + str(p_B[Q[mask]]))


                    info_gain[idx] += info_gain_B * p_q[i]

            print(info_gain)
            print(info_gain2)
            #pdb.set_trace()     
            return indicies[np.argmax(info_gain2)]

