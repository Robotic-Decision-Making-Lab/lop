# AbsBayesInfo.py
# Written Ian Rankin - July 2024
#
# A set of code to select a single active learning examples as an absloute query

import numpy as np
from scipy.integrate import quad_vec
from scipy.stats import beta

from lop.active_learning import ActiveLearner
from lop.models import PreferenceGP, PreferenceLinear
from lop.utilities import metropolis_hastings

import pdb

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


    ## integrand
    # The dq function that is to be integrated to solve the information gain
    # @param q - the query q value to be integrated over
    # @param args - the arguments passed to the integrand (p_b, all_f)
    #
    # @return
    def integrand(self, q, p_b, all_f, aa, bb, ml, integ_p_q_f_B):
        # [samples, f]
        p_q_f = beta.pdf(q, aa, bb)

        N = all_f.shape[1]

        # [samples, f, B]
        p_q_f_B = self.p_q_f_B_unorm(q, aa, bb, ml) / integ_p_q_f_B

        tmp = (p_q_f_B * p_b) / np.repeat(p_q_f[:,:,np.newaxis], p_q_f.shape[1], axis=2)
        result = p_q_f_B * np.log(tmp)
        result = np.where(np.isnan(result), 0, result)

        return result


    def p_q_f_B_unorm(self, q, aa, bb, ml):
        ml_r = np.repeat(ml[:,np.newaxis, :], ml.shape[1], axis=1)
        p_q_f = beta.pdf(q, aa, bb)
        p_q_f_B = np.where(q > ml_r, np.repeat(p_q_f[:,:,np.newaxis], p_q_f.shape[1], axis=2), 0)
        return p_q_f_B




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
        p_b, all_f = self.calc_one_time(candidate_pts[indicies], mu)

        ml = self.model.probits[2].mean_link(all_f)
        aa, bb = self.model.probits[2].get_alpha_beta_ml(all_f, ml)

        integ_p_q_f_B, err = quad_vec(self.p_q_f_B_unorm, 0,1, epsrel=0.001, workers=-1, limit=200, args=(aa,bb,ml))

        integ_r, err = quad_vec(self.integrand, 0, 1, epsrel=0.001, workers=-1, limit=200, args=(p_b, all_f, aa, bb, ml, integ_p_q_f_B))

        E_over_samples = np.sum(integ_r, axis=0) / self.M

        E_q_H_B = np.sum(-p_b * E_over_samples, axis=1)
        H_B = np.sum(-p_b * np.log(p_b))

        info_gain = H_B - E_q_H_B

        return indicies[np.argmax(info_gain)]



        





    