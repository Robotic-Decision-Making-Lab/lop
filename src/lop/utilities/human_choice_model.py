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

# human_choice_model.py
# Written Ian Rankin - March 2023
#
# A probability model of human decisions given estimated rewards of
# task.
# Using the Luce-Shepard choice rule.
# I'm using the standard formulation define in
# Asking Easy Questions: A User-Friendly Approach to Active Reward Learning (2019) 
#    E. Biyik, M. Palan, N.C. Landolfi, D.P. Losey, D. Sadigh


import numpy as np
from scipy import stats



## p_human_choice
# calculates the probability distribution for a set of estiamted rewards.
# Uses a softmax of the given rewards to model the distribution.
# @param r - the input reward vector (N,)
# @param p - [opt] peakiness of the human choice model, this can be tuned to set how flat or peaky
#              the probability distribution is.
#
# @return output probability distribution of human choice
def p_human_choice(r, p=1.0):
    r = r - np.max(r)
    e = np.exp(r*p) # exponent of r
    if len(r.shape) > 1:
        sum_e = np.sum(e,axis=-1)
        return e / sum_e[:,np.newaxis]
    else:
        return e / np.sum(e)

    



## sample_human_choice
# returns the index of the reward sampled given the probability distribution defined
# by the luce-shepard choice rule.
# @param r - the input reward vector (N,)
# @param p - [opt] peakiness of the human choice model, this can be tuned to set how flat or peaky
#              the probability distribution is.
# @param samples - [opt] sets how many different samples to take.
def sample_human_choice(r, p=1.0, samples=None):
    xk = np.arange(len(r))
    pdf = p_human_choice(r, p=p)

    print('pdf = ' + str(pdf))

    if samples is None:
        return np.random.choice(xk, p=pdf)
    else:
        return np.random.choice(xk, samples, p=pdf)







