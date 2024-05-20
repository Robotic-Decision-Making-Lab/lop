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

# mcmc_sampling.py
# Written Ian Rankin January 2024
#
# Monte-carlo markov chain sampling.
# Set of a functions to approximate sampling from a posterior distribution without
# the entire distribution defined.
#
# This is required for the sampling needed in the Mutual information learner 
# for linear models.
#
# Based on this medium article for the MCMC sampling
# https://exowanderer.medium.com/metropolis-hastings-mcmc-from-scratch-in-python-c21e53c485b7


import numpy as np

## normal_prop_dist
# A basic normal distribtion to use with the mcmc sampler
def normal_prop_dist(x, dimensions=1, std=2.0):
    return np.random.normal(x, std)


# metropolis_hastings
# A function to generate a set of samples from an unknown distribtuion
# using the metropolis-hastings algorithm.
# @param log_liklihood - the log_liklihood function
# @param num_samples - the number of samples to create
# @param dim - [opt, default 1] the number of dimmensions the sample distribution
# @param num_to_burn - [opt] the numer of samples to burn before saving them
# @param prop_distribution - [opt] the proposal distribution for the sampling
def metropolis_hastings(log_liklihood, 
                        num_samples,
                        dim = 1,
                        num_to_burn=500,
                        prop_distribution=normal_prop_dist):
    samples = []

    # get current state and liklihood
    cur_x = np.random.random(dim) * 2 - 1
    cur_ll = log_liklihood(cur_x)

    for i in range(num_samples + num_to_burn):
        # proposal state
        prop_x = prop_distribution(cur_x)
        
        prop_ll = log_liklihood(prop_x)
        accept_crit = np.exp(prop_ll - cur_ll)
        
        # uniform random between 0 and 1
        accept_thresh = np.random.random()
        
        # accept the proposal state as the current state
        if accept_crit > accept_thresh:
            cur_x = prop_x
            cur_ll = prop_ll
            
        
        if i >= num_to_burn:
            samples.append(cur_x)

    return np.array(samples)
