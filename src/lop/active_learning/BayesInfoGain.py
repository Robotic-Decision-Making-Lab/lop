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
from scipy.stats import multivariate_normal

import pdb








class BayesInfoGain(ActiveLearner):

    # def __init__(self):
    #     self.sample_pts = None


    ## p_B_pref_gp
    # Calculates the probabilaity of each pt in the given matrix as being the being the best path
    # but only does it for preference GPs
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param mu - a numpy array of mu values outputed from predict. numpy (n)
    def p_B_pref_gp(self, candidate_pts, mu):
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

            
            rv = multivariate_normal(mean=mu_star, cov=K_star_i)
            

            #x, y = np.mgrid[-1:1:.01, -1:1:.01]
            #pos = np.dstack((x, y))
            #plt.figure()
            #plt.contourf(x,y,rv.pdf(pos))
            
            zero_vector = np.zeros(len(candidate_pts)-1)
            p[i] = rv.cdf(zero_vector)

        print('p_sum = ' + str(np.sum(p)))
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

            # go through each possible query to calculate information gain on each
            for idx, Q_i in enumerate(indicies):
                
                Q = np.array(list(prev_selection) + [Q_i])

                if len(Q) < 2:
                    # THIS IS PROBABLY NOT THE RIGHT WAY TO HANDLE THIS
                    #return np.random.choice(indicies)
                    return np.argmax(mu)

                p_q = np.zeros(len(Q))
                for i, q in enumerate(Q):
                    mask = np.ones(len(Q), bool)
                    mask[i] = False

                    probit_Q = probit_mat[q,Q[mask]]
                    p_q[i] = np.prod(probit_Q)

                p_q = p_q / np.sum(p_q)

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
            #pdb.set_trace()     
            return indicies[np.argmax(info_gain)]

