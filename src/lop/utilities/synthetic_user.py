# synthetic_user.py
# Written Ian Rankin - August 2024
#
# A set of code to implement synthetic users

import numpy as np

from lop.utilities import gen_pairs_from_idx, \
                            sample_human_choice, p_human_choice, \
                            sample_unique_sets
from math import comb
from scipy.optimize import minimize_scalar

import pdb

def sigmoid(x):
    k = 4
    return 1 / (1 + np.exp(-k*x))

class SyntheticUser():

    ## constructor
    # @param fake_f - the fake function for the user.
    def __init__(self, fake_f):
        self.fake_f = fake_f

    ## choose
    # The function called to have the synthetic user choose a particular query rewards 
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return integer which query to select
    def choose(self, query_rewards):
        raise NotImplementedError("choose is not implemented")

    ## choose_pairs
    # The function called to have the synthetic user choose a particular query rewards
    # Identical to choose, but outputs the needed pairs to pass to the PreferenceModel add
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return pairs
    def choose_pairs(self, query_rewards):
        best_idx = self.choose(query_rewards)

        return gen_pairs_from_idx(best_idx, list(range(query_rewards.shape[0])))

    ## rate
    # The function called to have a synthetic user output a continous value [0,1] on a single path
    # @param query_rewards - numpy array (M,)
    #
    # @return float value of rating of query in bounds [0,1]
    def rate(self, query_reward):
        raise NotImplementedError("rate is not implemented")


class PerfectUser(SyntheticUser):

    ## choose
    # The function called to have the synthetic user choose a particular query rewards 
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return integer which query to select
    def choose(self, query_rewards):
        y = self.fake_f(query_rewards)

        return np.argmax(y)

    ## rate
    # The function called to have a synthetic user output a continous value [0,1] on a single path
    # @param query_rewards - numpy array (M,)
    #
    # @return float value of rating of query in bounds [0,1]
    def rate(self, query_reward):
        y = self.fake_f(query_reward)

        return sigmoid(y)
    




class HumanChoiceUser(SyntheticUser):

    ## constructor
    # @param fake_f - the fake function for the user.
    def __init__(self, fake_f, beta = 1.0, sigma = 1.0):
        super(HumanChoiceUser, self).__init__(fake_f)

        self.beta = beta
        self.sigma = sigma


    def kl_objective(self, b, desired_p, sample_queries):
        y = self.fake_f(sample_queries)
        p = p_human_choice(y, p=b)



        p_max = np.max(p, axis=1)
        
        
        p_max = np.where(p_max > 0.999999, 0.999999, p_max)
        p_max = np.where(p_max < 0.000001, 0.000001, p_max)
        
        # Minimize the matched KL divergence
        KL_1 = p_max * np.log(p_max / desired_p) + ((1 - p_max) * np.log((1- p_max) / (1 - desired_p)))

        tmp = p_max
        p_max = desired_p
        desired_p = tmp
        
        KL_2 = p_max * np.log(p_max / desired_p) + ((1 - p_max) * np.log((1- p_max) / (1 - desired_p)))

        return np.mean(KL_1+KL_2)


    def sampled_objective(self, b, desired_p, sample_queries):
        y = self.fake_f(sample_queries)
        p = p_human_choice(y, p=b)

        p_max = np.max(p, axis=1)

        pf = np.random.random(p_max.shape[0])

        count = np.sum(pf <= p_max)
        p_samp = count / sample_queries.shape[0]

        return (p_samp - desired_p)**2

    def rate_sampled_objective(self, sigma, desired_p, sample_queries, y):
        # [N x size_Query]
        rating = np.random.normal(y, scale=sigma)

        num_correct = np.sum(np.argmax(rating, axis=1) == np.argmax(y, axis=1))
        p_samp = (num_correct / y.shape[0])

        return (p_samp - desired_p)**2

    def sample_Qs(self, rewards, Q_size):
        num_Q = min(comb(rewards.shape[0], Q_size) * 0.5, 20000)
        if num_Q < 30:
            num_Q = min(30, comb(rewards.shape[0], Q_size))
        num_Q = int(num_Q)

        Qs = sample_unique_sets(rewards.shape[0], num_Q, Q_size)
        return Qs


    ## learn_beta
    # This function learns the required beta value and sigma for both 
    # pairwise and absloute queries
    # @param rewards - a set of reward values to try
    # @param p - the probability of selecting the best value
    # @param num_Q - [opt default=2] the number of points in each query
    def learn_beta(self, rewards, p, Q_size=2):
        Qs = self.sample_Qs(rewards, Q_size)

        self.learn_beta_pairwise(rewards, p, Q_size, Qs=Qs)
        self.learn_sigma(rewards, p, Q_size, Qs=Qs)


    ## learn_sigma
    # this function learns
    def learn_sigma(self, rewards, p, Q_size=2, Qs=None):
        if Qs is None:
            Qs = self.sample_Qs(rewards, Q_size)

        sample_Q = rewards[Qs]
        y = self.fake_f(sample_Q)

        res = minimize_scalar(self.rate_sampled_objective, bounds=[0.001, 10.0], args=(p, sample_Q, y), options={'xatol': 0.01})

        self.sigma = res.x


    ## learn_beta_pairwise
    # This function learns the required beta value
    # @param rewards - a set of reward values to try
    # @param p - the probability of selecting the best value
    # @param num_Q - [opt default=2] the number of points in each query
    def learn_beta_pairwise(self, rewards, p, Q_size=2, Qs=None):
        if Qs is None:
            Qs = self.sample_Qs(rewards, Q_size)

        sample_Q = rewards[Qs]

        res = minimize_scalar(self.sampled_objective, bounds=[0.01, 100.0], args=(p, sample_Q), options={'xatol': 0.01})

        self.beta = res.x



    def rate(self, query_rewards):
        y = self.fake_f(query_rewards)
        val = np.random.normal(loc=y, scale=self.sigma)

        return sigmoid(val)


    ## choose
    # The function called to have the synthetic user choose a particular query rewards 
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return integer which query to select
    def choose(self, query_rewards):
        y = self.fake_f(query_rewards)

        best_idx = sample_human_choice(y, p=self.beta)

        return best_idx


