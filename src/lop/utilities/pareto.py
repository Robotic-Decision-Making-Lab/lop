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
#
# pareto.py
# Written Ian Rankin - December 2023
#
# A set of utility functions for handling

import numpy as np



## get_pareto
# This function returns the indicies of the pareto optimal values in values
# Code taken directly from this stackoverflow post for the function
# is_pareto_efficient by user 'Peter'
# https://stackoverflow.com/questions/32791911/fast-calculation-of-pareto-front-in-python
# @param values - a numpy array of n values with k dimmensions numpy(n, k)
# 
# @return a list of indicies of each point in the pareto optimal set.
def get_pareto(values, return_mask=False):
    is_efficient = np.arange(values.shape[0], dtype=int)
    n_points = values.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(values):
        nondominated_point_mask = np.any(values>values[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        values = values[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient


