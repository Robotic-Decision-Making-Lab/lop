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

## k_fold_split
# A function to split the datasets for training the PreferenceGP
# @param y - the y data as [[(input data for probit)]]
# @param k - [opt, default=2] the number of data splits
def k_fold_split(y, k=2):
    data = []

    for i, y_data in enumerate(y):
        if y_data is not None:
            if i == 0:
                shuffle = np.arange(len(y_data))
                np.random.shuffle(shuffle)
                splits = np.array_split(shuffle, k)

                # [probits[[split1, split2]], ...]
                split_data = [[y_data[idx] for idx in split] for split in splits]
                
            elif i == 1 or i == 2:
                shuffle = np.arange(len(y_data[0]))
                np.random.shuffle(shuffle)
                splits = np.array_split(shuffle, k)

                split_data = [[[y_data[0][idx], y_data[1][idx]] for idx in split] for split in splits]
            else:
                raise Exception("Not sure how to handle this split.")

            data.append(split_data)
            
        else:
            data.append(None)


    actual_splits = []
    for j in range(k):
        split = []

        for i in range(len(y)):
            if data[i] is not None and len(data[i][j]) != 0:
                if i == 0:
                    split += [np.array(data[i][j])]
                elif i == 1 or i == 2:
                    arr0 = np.array([val[0] for val in data[i][j]])
                    arr1 = np.array([val[1] for val in data[i][j]])
                    split += [(arr0, arr1)]
            else:
                split += [None]

        actual_splits.append(split)

    return actual_splits

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

def union_splits(splits, idx_to_combine):
    # TODO This needs to be fixed
    return splits[idx_to_combine[0]]
    
    
    uni = []
    for j in range(len(splits[0])):
        if splits[0][j] is None:
            uni.append(None)
        elif isinstance(splits[0][j], np.ndarray):
            if len(splits[0][j].shape) == 2:
                uni.append(np.zeros((0,splits[0][j].shape[1])))
            else:
                raise Exception("Can't handle this union split with nd array")
        elif isinstance(splits[0][j], tuple):
            uni.append((np.array([]), np.array([])))
        else:
            pdb.set_trace()
            raise Exception("Can't handle this split")

    
    
    for i in range(len(splits)):
        if uni[i] is not None:
            for j in idx_to_combine:
                if splits[i][j] is None:
                    pass
                elif i == 0:
                    uni[i] = np.append(uni[i], splits[i][j], axis=0)
                elif i == 1 or i == 2:
                    uni[i] = (  np.append(uni[i][0], splits[j][i][0], axis=0), \
                                np.append(uni[i][1], splits[j][i][1], axis=0))
    if uni[1] is not None and len(uni[1][0]) == 0:
        uni[1] = None
    if uni[2] is not None and len(uni[2][0]) == 0:
        uni[2] = None

    return uni

    