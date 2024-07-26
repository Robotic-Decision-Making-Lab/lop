# BayesInfoGain.py
# Written Ian Rankin - May 2024
#
# Take 2 on the Bayes info gain. Trying to see if I recalculate the info gain.
# Maybe this time it'll have positive information gain. lol

import numpy as np
from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, GP, PreferenceLinear
from lop.utilities import calc_cdf


from copy import deepcopy

import pdb

MIN_LOG_VALUE = 1e-17

class BayesInfoGain(ActiveLearner):

    ## Constructor
    # @param default_to_pareto - [opt default=False] sets whether to always assume
    #               prefering pareto optimal choices when selecting points, if not particulary told not to
    # @param alaways_select_best - [opt default=False] sets whether the select function should append the
    #               the top solution to the front of the solution set every time.
    # @param p_q_B_method - [opt defautl='probit'] the method to calculate the p_q given B Options are:
    #                    ['probit', '999', '99']
    def __init__(self, default_to_pareto=False, always_select_best=False, p_q_B_method='probit'):
        super(BayesInfoGain, self).__init__(default_to_pareto, always_select_best)
        # this just forces the object to fail if approxcdf is not installed
        import approxcdf
        self.p_q_B_method = p_q_B_method

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
            sub_probit = probit_mat[q_i, Q_subi]
            p_q[i] = np.prod(sub_probit)

            # Calculate p_q_B for each B
            for B_i in range(N):
                if B_i not in Q:
                    p_q_B[i, B_i] = p_q[i]#np.prod(probit_mat[q_i, Q_subi])
                else:
                    # Set what p_q_B is
                    if self.p_q_B_method == 'probit':
                        sub_probit = probit_mat[q_i, Q_subi]
                        sub_probit_w = np.where(q_i == B_i, np.fmax(sub_probit, 1-sub_probit), sub_probit)
                        p_q_B[i, B_i] = np.prod(sub_probit)
                    elif self.p_q_B_method == '999': 
                        p_q_B[i, B_i] = np.prod(np.where(q_i == B_i, 0.999, sub_probit))
                    elif self.p_q_B_method == '99':
                        p_q_B[i, B_i] = np.prod(np.where(q_i == B_i, 0.99, sub_probit))

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


    def get_p_B_probit(self, candidate_pts, mu):
        if self.first_call_greedy:
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
            self.p_B = p_B
            self.probit_mat = probit_mat
            self.first_call_greedy = False
        else:
            p_B = self.p_B
            probit_mat = self.probit_mat

        return p_B, probit_mat

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
    def select_greedy(self, candidate_pts, mu, data, indicies, prev_selection, debug=False):
        N = len(mu)
        indicies = list(indicies)

        # If there is no data, just provide a random selection
        if self.model.X_train is None:
            return np.random.choice(list(indicies), 1)[0]

        if debug:
            print('mu = ' + str(mu))

        p_B, probit_mat = self.get_p_B_probit(candidate_pts, mu)
        
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

        return indicies[np.argmax(info_gain)] #indicies[np.argmax(info_gain[indicies])]               
                


    def select_pair(self, candidate_pts, mu, data, indicies, prev_selection, debug=False):
        p_B, probit_mat = self.get_p_B_probit(candidate_pts, mu)

        if self.model.X_train is None:
            idxs = np.random.choice(list(indicies), 2)
            return (idxs[0], idxs[1])

        if debug:
            print('\t p_B: ' + str(p_B))

        N = len(p_B)
        
        # Calculate the current entropy
        H_B = -np.sum(np.where(p_B < MIN_LOG_VALUE, 0, p_B * np.log(p_B)))

        p_q = probit_mat
        if debug:
            print('\t p_q: ')
            print(p_q)


        # define p_q_B
        # This probably needs to be significantly updated
        p_q_B = np.repeat(p_q[:,:,np.newaxis], N, axis=2)
        for i in range(N):
            if self.p_q_B_method == 'probit':
                p_q_B[i,:,i] = np.fmax(p_q[i,:], p_q[:,i])
                p_q_B[:,i,i] = np.fmin(p_q[i,:], p_q[:,i])
            elif self.p_q_B_method == '999':
                p_q_B[i,:,i] = 0.999
                p_q_B[:,i,i] = 0.001
            elif self.p_q_B_method == '99':
                p_q_B[i,:,i] = 0.99
                p_q_B[:,i,i] = 0.01

            #p_q_B[i,i,i] = 0.99

            # Same query is always going to be probability of 0.5
            p_q_B[i,i,:] = 0.5
    
        if debug:
            print('\tp_q_B:')
            for i in range(N):
                print(p_q_B[:,:,i])
            

        p_B_rep = np.repeat(np.repeat(p_B[np.newaxis,np.newaxis, :], N, axis=0), N, axis=1)
        p_q_rep = np.repeat(p_q[:,:,np.newaxis], N, axis=2)
        p_B_q = p_q_B * p_B_rep / p_q_rep


        p_B_q = p_B_q / np.repeat(np.sum(p_B_q, axis=2)[:,:,np.newaxis], N, axis=2)

        if debug:
            print('\tp_B_q:')
            for i in range(N):
                print(p_B_q[:,:,i])


        # Calculate the predicted post entropy
        H_B_q = -np.sum(np.where(p_B_q < MIN_LOG_VALUE, 0, p_B_q * np.log(p_B_q)), axis=2)

        if debug:
            print('\t H_B_q: ')
            print(H_B_q)

        H_B_Q = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                H_B_Q[i,j] = p_q[i,j] * H_B_q[i,j] + p_q[j,i] * H_B_q[j,i]

        if debug:
            print('\t H_B_Q: ')
            print(H_B_Q)

        # calculate the information gain for a query Q
        info_gain = H_B - H_B_Q

        if debug:
            print('\tInfo gain')
            print(info_gain)


        idx_best = self.pick_pair_from_metric(info_gain, prev_selection)
        

        print('idx_best: ' + str(idx_best))


        return idx_best




