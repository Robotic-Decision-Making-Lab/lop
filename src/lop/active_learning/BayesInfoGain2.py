# BayesInfoGain2.py
# Written Ian Rankin - May 2024
#
# Take 2 on the Bayes info gain. Trying to see if I recalculate the info gain.
# Maybe this time it'll have positive information gain. lol

import numpy as np
from lop.active_learning import ActiveLearner, BayesInfoGain
from lop.models import PreferenceGP, GP, PreferenceLinear
from lop.utilities import calc_cdf


from copy import deepcopy

import pdb

MIN_LOG_VALUE = 1e-17

class BayesInfoGain2(BayesInfoGain):

    ## calc_H_B_Q
    # Calculate the expected entropy of B given Q.
    # H(B|Y,Q)
    def calc_H_B_Q(self, Q, p_B, probit_mat, debug=False):
        N = len(p_B)

        # calculate p_q given Y
        p_q = np.ones(len(Q))
        # p_q given Y and B
        p_q_B = np.ones((len(Q), N)) # [q, B_i]
        for i, q_i in enumerate(Q):
            Q_subi = np.copy(Q)
            Q_subi = np.delete(Q_subi, i)
            p_q[i] = np.prod(probit_mat[q_i, Q_subi])

            # Calculate p_q_B for each B
            for B_i in range(N):
                if B_i not in Q:
                    p_q_B[i, B_i] = np.prod(probit_mat[q_i, Q_subi])
                else:
                    # Set what p_q_B is 
                    p_q_B[i, B_i] = np.prod(np.where(q_i == B_i, 0.999, 0.001))

        p_q = p_q / np.sum(p_q)
        p_q_B = p_q_B / np.sum(p_q_B, axis=0)
        if debug:
            print('\tp_q = ' + str(p_q))
            print('\tp_q_B = ')
            print(p_q_B)



        # calculate probability of B given Y and each q using Bayes rule
        p_B_q = p_q_B * p_B / np.repeat(p_q[:,np.newaxis], N, axis=1)
        p_B_q = p_B_q / np.repeat(np.sum(p_B_q, axis=1)[:, np.newaxis], N, axis=1) 

        if debug:
            print('\tp_B_q = ' + str(p_B_q))
        

        # Calculate the predicted post entropy
        H_B_q = -np.sum(np.where(p_B_q < MIN_LOG_VALUE, 0, p_B_q * np.log(p_B_q)), axis=1)

        if debug:
            print('\tH(B|Y, q) = ' + str(H_B_q))

        H_B_Q = np.sum(p_q * H_B_q)

        if debug:
            print('\tE_q[H(B|Y, q)] = H(B|Y,Q) = ' + str(H_B_Q))
        
        return H_B_Q

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
    def select_greedy(self, candidate_pts, mu, data, indicies, prev_selection, debug=True):
        N = len(mu)
        indicies = list(indicies)

        if debug:
            print('mu = ' + str(mu))

        if isinstance(self.model, PreferenceGP):
            # Get the probabilities of which candidate_pts is the best.
            p_B = self.p_B_pref_gp(candidate_pts, mu)
            # precalculate the probit between each candidate_pts
            probit_mat = self.model.probits[0].likelihood_all_pairs(mu)
        elif isinstance(self.model, PreferenceLinear):
            # precalculate the probit between each candidate_pts
            probit_mat = self.model.probits[0].likelihood_all_pairs(mu)

            # Get the probabilities of which candidate_pts is the best.
            p_B = self.p_B_pref_linear(candidate_pts, mu, probit_mat)
        
        
        if debug:
            print('p_B = ' + str(p_B))


        # Calculate the current entropy
        H_B = -np.sum(np.where(p_B < MIN_LOG_VALUE, 0, p_B * np.log(p_B)))

        if debug:
            print('\tH(B|Y) = ' + str(H_B))

        info_gain = np.zeros(len(indicies))

        # Calculate the info gain for each sample
        for idx, Q_i in enumerate(indicies):
            Q = np.array(list(prev_selection) + [Q_i])
            if debug:
                print('Q_i: ' + str(Q_i) + ' Q: ' + str(Q))

            if len(Q) < 2:
                # THIS IS PROBABLY NOT THE RIGHT WAY TO HANDLE THIS
                return np.random.choice(indicies)
                #return np.argmax(p_B)

            H_B_Q = self.calc_H_B_Q(Q, p_B, probit_mat, debug)

            info_gain[idx] = H_B - H_B_Q

        if debug:
            print('Info gain = ' + str(info_gain))

        return indicies[np.argmax(info_gain)]               
                





