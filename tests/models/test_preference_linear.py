# test_GP.py
# Written Ian Rankin - December 2023
#
#

import pytest
import lop

import numpy as np


def f_sq(x, data=None):
    return (x/10.0)**2

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def f_lin(x, data=None):
    #return x[:,0]*x[:,1]
    return x[:,0]+x[:,1]

def test_pref_linear_construction():
    gp = lop.PreferenceLinear(lop.RBF_kern(1.0, 1.0))

    assert gp is not None

def test_pref_linear_not_optimized():
    pm = lop.PreferenceLinear()
    X_train = np.array([[0,0],[1,2],[2,4],[3,2],[4.2, 5.6],[6,2],[7,8]])


    y,_ = pm.predict(X_train)

    assert not np.isnan(y).any()


def test_pref_linear_function():
    np.random.seed(0)
    pm = lop.PreferenceLinear()

    X_train = np.array([[0,0],[1,2],[2,4],[3,2],[4.2, 5.6],[6,2],[7,8]])
    pairs = lop.generate_fake_pairs(X_train, f_lin, 0) + \
            lop.generate_fake_pairs(X_train, f_lin, 1) + \
            lop.generate_fake_pairs(X_train, f_lin, 2) + \
            lop.generate_fake_pairs(X_train, f_lin, 3) + \
            lop.generate_fake_pairs(X_train, f_lin, 4)

    pm.add(X_train, pairs)
    pm.optimize()

    assert pm is not None
    assert pm.optimized
    assert pm.n_loops > 0 and pm.n_loops < 101

    y, _ = pm.predict(X_train)

    for i in range(len(X_train)):
        if i != 6:
            assert y[6] > y[i]
        if i!= 0:
            assert y[0] < y[i]

def test_pref_linear_function_2_way_pair():
    pm = lop.PreferenceLinear()

    X_train = np.array([[0,0],[1,2]])
    pairs = lop.generate_fake_pairs(X_train, f_lin, 0)

    pm.add(X_train, pairs)
    pm.optimize()

    assert pm is not None
    assert pm.optimized
    assert pm.n_loops > 0 and pm.n_loops < 15

    y, _ = pm.predict(X_train)

    assert not np.isnan(y).any()

