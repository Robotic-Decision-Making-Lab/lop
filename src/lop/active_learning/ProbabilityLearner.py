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

from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear
from lop.utilities import metropolis_hastings

from scipy.stats import multivariate_normal

import matplotlib.pyplot as plt
import pdb

class ProbabilityLearner(ActiveLearner):
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
        if isinstance(self.model, PreferenceGP):
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
                #K_star_i += np.ones((len(K_star_i), len(K_star_i))) * 2 * sig * sig

                mu_star = mu[idx_i] - mu[i]

                rv = multivariate_normal(mean=mu_star, cov=K_star_i)

                x, y = np.mgrid[-1:1:.01, -1:1:.01]
                pos = np.dstack((x, y))
                plt.figure()
                plt.contourf(x,y,rv.pdf(pos))
                

                p[i] = rv.cdf([0,0])

        p = p / np.sum(p)

        print(candidate_pts)
        print(mu)
        print(p)

        plt.show()
        pdb.set_trace()


                




        


