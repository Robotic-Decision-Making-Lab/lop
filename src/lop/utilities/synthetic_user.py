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

def sigmoid(x, k=4, b=0):
    return 1 / (1 + np.exp(-(k*x+b)))

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
    def __init__(self, fake_f, beta = 1.0, sigma = 1.0, bot_range=0.1, top_range=0.9):
        super(HumanChoiceUser, self).__init__(fake_f)

        self.beta = beta
        self.sigma = sigma

        self.bot_range = bot_range
        self.top_range = top_range

        self.k = 4.0
        self.b = 0.0


    def kl_objective(self, b, desired_p, sample_queries):
        y = (self.fake_f(sample_queries) * self.k) + self.b
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


    def sampled_objective(self, b, desired_p, sample_queries, y):
        p = p_human_choice(y, p=b)

        p_max = np.max(p, axis=1)

        pf = np.random.random(p_max.shape[0])

        count = np.sum(pf <= p_max)
        p_samp = count / sample_queries.shape[0]

        return (p_samp - desired_p)**2

    def sampled_objective2(self, b, desired_p, rewards, Q_size, Qs=None):
        if Qs is None:
            Qs = self.sample_Qs(rewards, Q_size)

        y = (self.fake_f(rewards[Qs]) * self.k) + self.b

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
        num_Q = min(comb(rewards.shape[0], Q_size) * 0.5, 60000)
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
    def learn_beta(self, rewards, p, Q_size=2, p_sigma=None):
        Qs = self.sample_Qs(rewards, Q_size)

        ### before training everything else, set sigmoid such that 
        # the ranges are about the same across functions
        min_sigmoid = np.log(self.bot_range) - np.log(1 - self.bot_range)
        max_sigmoid = np.log(self.top_range) - np.log(1 - self.top_range)
        
        y = self.fake_f(rewards)
        f_max = np.max(y)
        f_min = np.min(y)

        # learn linear equation so min and max sigmoid are mapped to the highest and lowest points
        self.k = (max_sigmoid - min_sigmoid) / (f_max - f_min)
        self.b = min_sigmoid - self.k * f_min

        print('k = ' + str(self.k) + ' b = ' + str(self.b))

        if p_sigma is None:
            p_sigma = p


        self.learn_beta_pairwise(rewards, p, Q_size, Qs=Qs)
        self.learn_sigma(rewards, p_sigma, Q_size, Qs=Qs)
        print('beta = ' + str(self.beta) + ' sigma=' + str(self.sigma))


    ## learn_sigma
    # this function learns
    def learn_sigma(self, rewards, p, Q_size=2, Qs=None):
        if Qs is None:
            Qs = self.sample_Qs(rewards, Q_size)

        sample_Q = rewards[Qs]
        y = (self.fake_f(sample_Q) * self.k) + self.b

        for i in range(10):
            res = minimize_scalar(self.rate_sampled_objective, bounds=[0.0001, 2.0], args=(p, sample_Q, y), options={'xatol': 0.01})
        
            if res.fun < 0.02:
                break
        if res.fun > 0.02:
            print('Bad setting of sigma value after 10 tries. Throwing exception')
            raise Exception("Bad setting of sigma values after 10 tries.")

        print('Tuned sigma with error =' + str(res.fun))
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

        y = (self.fake_f(sample_Q) * self.k) + self.b

        for i in range(10):
            res = minimize_scalar(self.sampled_objective, bounds=[0.01, 150.0], args=(p, sample_Q, y), options={'xatol': 0.01})
            #res = minimize_scalar(self.sampled_objective2, bounds=[0.01, 150.0], args=(p, rewards, Q_size), options={'xatol': 0.01})


            if res.fun < 0.02:
                break
        if res.fun > 0.02:
            print('Bad setting of beta value after 10 tries. This probably means the set is inseperable. Throwing excpetion')
            raise Exception("Bad setting of beta values after 10 tries. This probably means the evaluation set is inseperable")

        print('Tuned beta with error =' + str(res.fun))
       
        self.beta = res.x



    def rate(self, query_rewards):
        y = (self.fake_f(query_rewards) * self.k) + self.b
        val = np.random.normal(loc=y, scale=self.sigma)

        return sigmoid(val, k=1, b=0)


    ## choose
    # The function called to have the synthetic user choose a particular query rewards 
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return integer which query to select
    def choose(self, query_rewards):
        y = (self.fake_f(query_rewards) * self.k) + self.b

        best_idx = sample_human_choice(y, p=self.beta)

        return best_idx


