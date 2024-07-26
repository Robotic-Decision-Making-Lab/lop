# AbsBayesInfo.py
# Written Ian Rankin - July 2024
#
# A set of code to select a single active learning examples as an absloute query

import numpy as np

from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, PreferenceLinear

from lop.utilities import metropolis_hastings

class AbsBayesInfo(ActiveLearner):

    ## constructor
    # @param
    def __init__(self, M=100, default_to_pareto=False, always_select_best=False):
        super(AbsBayesInfo, self).__init__(default_to_pareto,always_select_best)
        self.M = M
        

    def calc_one_time(self, candidate_pts, mu):
        if self.first_call_greedy:
            if isinstance(self.model, PreferenceGP):
                # Get the probabilities of which candidate_pts is the best.
                p_B = self.p_B_pref_gp(candidate_pts, mu)
                
                cov = self.model.cov

                all_Q = np.random.multivariate_normal(mu, cov, size=self.M)
            elif isinstance(self.model, PreferenceLinear):
                w_samples = metropolis_hastings(self.model.loss_func, self.M, dim=candidate_pts.shape[1])

                w_norm = np.linalg.norm(w_samples, axis=1)
                w_samples = w_samples / np.tile(w_norm, (2,1)).T
                # generate possible outputs from weighted samples
                all_Q = (candidate_pts @ w_samples.T).T

                # Get the probabilities of which candidate_pts is the best.
                p_B = self.p_B_pref_linear(candidate_pts, mu)
            self.p_B = p_B
            self.all_Q = all_Q
            self.first_call_greedy = False
        else:
            p_B = self.p_B
            all_Q = self.all_Q

        return p_B, all_Q


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

        # If there is no data, just provide a random selection
        if self.model.X_train is None:
            return np.random.choice(list(indicies), 1)[0]

        # Calculate p_B
        # all_f = [samples, num_candidate_pts]
        p_b, all_f = self.calc_one_time(candidate_pts, mu)

        self.model.probits[1]??


        





    