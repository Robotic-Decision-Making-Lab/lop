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

def union_splits(splits, idx_to_combine):
    import pdb
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

    