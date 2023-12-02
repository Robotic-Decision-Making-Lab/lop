# test_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities used by the lop algorithms.

import pytest
import pdb

import lop
import numpy as np

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def f_lin(x, data=None):
    #return x[:,0]*x[:,1]
    return x

def f_sq(x, data=None):
    return x[:,0]*x[:,0] + 1.2*x[:,1]


# def test_fake_pairs():
#     X_train = np.array([0,1,2,3])
#     pairs = lop.generate_fake_pairs(X_train, f_lin, 0)

#     for i, p in enumerate(pairs):
#         assert p[0] == 1
#         assert p[1] == 0
#         assert p[2] == i+1

    
def test_the_test():
    assert True

