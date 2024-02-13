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
# training_utility.py
# Written Ian Rankin - October 2021
#
# A set of utility functions for training.

import numpy as np
import math
import random
import pdb


## k_fold_x_y
# k fold split for both x and y training data
# @param X - the X input data
# @param y - the y training data
# @param k - the max k to split
#
# @return list of indicies
def k_fold_x_y(X, y, k):
    N = X.shape[0]

    # check to make sure 
    if k > int(N / 2.0):
        k = int(N / 2.0)
        if k < 2:
            return None
    set_max_size = math.ceil(N / k)

    splits = [[] for i in range(k)]
    conn = [set() for i in range(N)]

    selected = set()
    remaining = list(range(N))

    if y[0] is not None:
        for pair in y[0]:
            conn[pair[1]].add(pair[2])
            conn[pair[2]].add(pair[1])

    

    conn = [list(conn[i]) for i in range(N)]

    # Select an element
    while len(remaining) > 0:
        idx = remaining[np.random.randint(0, len(remaining))]
        remaining.remove(idx)
        selected.add(idx)


        # choose split
        sizes = [len(splits[i]) for i in range(k)]
        split_idx = np.argmin(np.array(sizes))

        splits[split_idx].append(idx)

        random.shuffle(conn[idx])
        shuffled_idxs = conn[idx]

        # try to add as many as possible while still allowing all k fold to even
        for shuf_idx in shuffled_idxs:
            if len(splits[split_idx]) >= set_max_size:
                break

            if shuf_idx not in selected:
                splits[split_idx].append(shuf_idx)
                selected.add(shuf_idx)
                remaining.remove(shuf_idx)

    for split in splits:
        split.sort()

    return splits

def get_y_with_idx(y, indicies):
    idx_set = set(indicies)

    idx_mapping = {}
    for i in range(len(indicies)):
        idx_mapping[indicies[i]] = i

    y_new = [None, None, None]

    if y[0] is not None:
        for pair in y[0]:
            if pair[1] in idx_set and pair[2] in idx_set:
                if y_new[0] is None:
                    y_new[0] = []
                y_new[0].append(np.array([pair[0], idx_mapping[pair[1]], idx_mapping[pair[2]]]))

        if y_new[0] is not None:
            y_new[0] = np.array(y_new[0])

    
    if y[1] is not None:
        y_new_0 = []
        y_new_1 = []
        for i in range(len(y[1][0])):
            if y[1][1][i] in idx_set:
                y_new_0.append(y[1][0][i])
                y_new_1.append(idx_mapping[y[1][1][i]])

        if len(y_new_1) > 0:
            y_new[1] = (np.array(y_new_0), np.array(y_new_1))

    if y[2] is not None:
        y_new_0 = []
        y_new_1 = []
        for i in range(len(y[2][0])):
            if y[2][1][i] in idx_set:
                y_new_0.append(y[2][0][i])
                y_new_1.append(idx_mapping[y[2][1][i]])

        if len(y_new_1) > 0:
            y_new[2] = (np.array(y_new_0), np.array(y_new_1))


    return y_new

