# synthetic_user.py
# Written Ian Rankin - August 2024
#
# A set of code to implement synthetic users

import numpy as np

from lop.utilities import gen_pairs_from_idx, sample_human_choice

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
    def __init__(self, fake_f, beta = 1.0):
        super(HumanChoiceUser, self).__init__(fake_f)

        self.beta = beta


    ## learn_beta
    # This function learns the required beta value
    # @param rewards - a set of reward values to try
    # @param p - the probability of selecting the best value
    # @param num_Q - [opt default=2] the number of points in each query
    def learn_beta(self, rewards, p, num_Q=2):
        pass
        # TODO

    ## choose
    # The function called to have the synthetic user choose a particular query rewards 
    # @param query_rewards - numpy array of rewards for a query (N x M), N paths, M-dimm reward
    #
    # @return integer which query to select
    def choose(self, query_rewards):
        y = self.fake_f(query_rewards)

        best_idx = sample_human_choice(y, p=self.beta)

        return np.argmax(y)


