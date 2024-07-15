# sample_utility.py
# Writen Ian Rankin - July 2024
#
# This is a set of function that are helper functions for sampling.
# Basically a set of code that might be helpful to be reusable, but
# don't fit anywhere else.

import numpy as np

def sample_nonunique_sets(s, num_sets, size_set):
    out = np.empty((num_sets, size_set))

    for i in range(num_sets):
        out[i] = np.random.choice(s, size_set, replace=False)

    return out


# @param s - the set to select from (if int treated as np.arange(s))
# @param num_sets - the number of unique sets of size set_size to return
# @param set_size - the size of the sets to return. Must be smaller than len(s)
def sample_unique_sets_fast(s, num_sets, size_set):
    num_unique = 0
    all_uniq = np.empty((0, size_set))

    while num_unique < num_sets:
        # defines whether to oversample, or exactly sample
        to_sample = int((num_sets - num_unique) * 1.2)
        new_samp = sample_nonunique_sets(s, to_sample, size_set)

        # find non_unique samples in new_samp
        new_samp.sort(axis=1)

        all_uniq = np.append(all_uniq, new_samp, axis=0)
        all_uniq = np.unique(all_uniq, axis=0)

        num_unique = all_uniq.shape[0]

    return all_uniq[:num_sets,:]




## sample unique sets
# Wrapper to make it easier to use.
# @param s - the set to select from (if int treated as np.arange(s))
# @param num_sets - the number of unique sets of size set_size to return
# @param set_size - the size of the sets to return. Must be smaller than len(s)
#
# @return a numpy array of 
def sample_unique_sets(s, num_sets, size_set):
    if isinstance(s, int):
        s_len = s
        s = np.arange(s_len, dtype=int)
    else:
        s_len = len(s)

    return sample_unique_sets_fast(s, num_sets, size_set)
    