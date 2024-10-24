# Copyright 2021 Ian Rankin
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
#
# preference_pairs.py
# Written Ian Rankin - October 2021
#
# A set of utility functions for handling preference pairs.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence
import pdb


## get_df
# This function returns the dk of the preference pair.
# It's a helper function to encode preferences as a tuple (dk, index1, index2)
# @param u - the output value of the first input
# @param v - output value of the second input
def get_dk(u, v):
    #if (isinstance(u, (int, float))) or (not isinstance(v, (int, float))):
    if isinstance(u, (Sequence, np.ndarray)) or isinstance(v, (Sequence, np.ndarray)):
        raise TypeError("get_dk was not passed a scalar value")
    if u > v:
        return -1
    elif u < v:
        return 1
    else:
        return -1 # probably handle it this way... I could also probably just return 0

## preference
# a function that generates a preference between two indicies
# @param u_idx - the index of the larger point
# @param v_idx - the index of the smaller point
#
# @return the preference pair (-1, u_idx, v_idx)
def preference(u_idx,v_idx):
    return (get_dk(1,0), u_idx, v_idx)

## gen_pairs_from_idx
# This function is given the best index selected from a user selection
# and generates the pairs needed to be passed to a preference GP
# @param best_idx - the index that was determined to be best of the given indicies
# @param indicies - the list of indicies the best_idx is better than.
#                   best_idx is allowed to be indicies without breaking anything
#
# @return - list of pairs [(dk, uk, vk), ...]
def gen_pairs_from_idx(best_idx, indicies):
    pairs = []
    for idx in indicies:
        if idx != best_idx:
            pairs.append(preference(best_idx, idx))

    return pairs

## ranked_pairs_from_fake
# generates a all of the ranked pairs from fake inputs
#
def ranked_pairs_from_fake(X, fake_f):
    y = fake_f(X)

    y_sorted_idx = np.argsort(y)

    pairs = []
    for i in range(len(y)):
        for j in range(i+1, len(y)):
            pairs.append((get_dk(y[i], y[j]), i, j))

    return pairs

## generate_fake_pairs
# generates a set of pairs of data from faked data
# helper function for fake input data
# @param X - the inputs to the function
# @param real_f - the real function to estimate
def generate_fake_pairs(X, real_f, pair_i, data=None):
    Y = real_f(X, data=data)

    pairs = [(get_dk(Y[pair_i], y),pair_i, i) for i, y in enumerate(Y) if i != pair_i]
    return pairs

## generate_ranking_pairs
# generate a set of ranking pairs from real fake data.
# @param X - the inputs to the function
# @param real_f - the real function to estimate.
def generate_ranking_pairs(X, real_f):
    real_y = real_f(X)

    sorted_idx = np.argsort(real_y)
    sorted_idx = sorted_idx[::-1]

    pairs = []
    for i, idx1 in enumerate(sorted_idx[:-1]):
        for j, idx2 in enumerate(sorted_idx[i+1:]):
            pairs.append((get_dk(1,0), idx1, idx2))



