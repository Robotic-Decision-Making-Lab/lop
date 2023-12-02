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


## k_fold_split
# A function to split the datasets for training the PreferenceGP
# @param y - the y data as [[(input data for probit)]]
def k_fold_half(y):
    data = []

    for y_data in y:
        shuffle = np.arange(len(y_data))
        np.random.shuffle(shuffle)

        splits = np.array_split(shuffle, 2)

        # [probits[[split1, split2]], ...]
        split_data = [[y_data[idx] for idx in split] for split in splits]

        data.append(split_data)

    actual_splits = []
    for j in range(2):
        split = []

        for i in range(len(y)):
            split += [np.array(data[i][j])]


        actual_splits.append(split)

    return actual_splits

