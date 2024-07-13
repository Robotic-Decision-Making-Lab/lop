# test_fake_function.py
# Written Ian Rankin - December 2023
#
# A set of tests of various fake functions used for experiments with lop.

import pytest
import pdb

import lop
import numpy as np


def test_sample_utility_small():
    s = lop.sample_unique_sets(5,3,2)

    assert s.shape[0] == 3
    assert s.shape[1] == 2

    # check if valid sets
    for i in range(s.shape[0]):
        uni, counts = np.unique(s[i], return_counts=True)
        assert (counts == 1).all()

    # check if each set is valid
    s_sorted = np.sort(s, axis=1)
    for i in range(s.shape[0]):
        for j in range(j, s.shape[0]):
            assert not (s_sorted[i] == s_sorted[j]).all()

