# test_fake_function.py
# Written Ian Rankin - December 2023
#
# A set of tests of various fake functions used for experiments with lop.

import pytest
import pdb

import lop
import numpy as np



def test_nonunique_sample():
    s = lop.sample_nonunique_sets(5,10,2)

    assert s.shape[0] == 10
    assert s.shape[1] == 2

    # check if valid sets
    for i in range(s.shape[0]):
        uni, counts = np.unique(s[i], return_counts=True)
        assert (counts == 1).all()




def test_sample_unique_sets_small():
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
        for j in range(i+1, s.shape[0]):
            assert not (s_sorted[i] == s_sorted[j]).all()



def test_sample_unique_sets_ensure_correct():
    for itr in range(50):
        s = lop.sample_unique_sets(5,8,2)

        assert s.shape[0] == 8
        assert s.shape[1] == 2

        # check if valid sets
        for i in range(s.shape[0]):
            uni, counts = np.unique(s[i], return_counts=True)
            assert (counts == 1).all()

        # check if each set is valid
        s_sorted = np.sort(s, axis=1)
        for i in range(s.shape[0]):
            for j in range(i+1, s.shape[0]):
                assert not (s_sorted[i] == s_sorted[j]).all()



def test_sample_unique_sets_large():
    s = lop.sample_unique_sets(300,1000,6)

    assert s.shape[0] == 1000
    assert s.shape[1] == 6

    # check if valid sets
    for i in range(s.shape[0]):
        uni, counts = np.unique(s[i], return_counts=True)
        assert (counts == 1).all()

    # check if each set is valid
    s_sorted = np.sort(s, axis=1)
    for i in range(s.shape[0]):
        for j in range(i+1, s.shape[0]):
            assert not (s_sorted[i] == s_sorted[j]).all()