# test_hyperparameter_search.py
# Written Ian Rankin - January 2024
#
# A set of tests investigating hyperparameter optimization for preference GP's

import pytest
import numpy as np
import pdb

import lop

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_hyperparameter_search_does_not_crash():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))


    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=True)

    assert gp is not None


def test_get_hyperparameters_multiple_y_trains():
    m = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))

    p = m.get_hyper()
    assert p[0] == 0.5 and p[1] == 0.7

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    m.add(X_train, pairs)

    p = m.get_hyper()
    assert p[0] == 0.5 and p[1] == 0.5 and p[2] == 0.7

    X_abs = np.array([1.5, 2.5])
    y_abs = f_sin(X_abs)

    m.add(X_abs, y_abs, type='abs')

    p = m.get_hyper()
    assert p[0] == 0.5 and p[3] == 0.5 and p[4] == 0.7

def test_set_hyperparameters_multiple_y_trains():
    m = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))


    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    m.add(X_train, pairs)


    X_abs = np.array([1.5, 2.5])
    y_abs = f_sin(X_abs)

    m.add(X_abs, y_abs, type='abs')

    p_set = np.array([0.4,0.3,0.2,0.6,1.2])
    m.set_hyper(p_set)
    p = m.get_hyper()
    assert (p == p_set).all()



def test_hyperparameter_search_somewhat_converges():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))


    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=True)


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()


    y, sigma = gp.predict(X_train)

    for i in range(len(X_train)):
        if i != 0:
            assert y[0] > y[i]
        if i!= 1:
            assert y[1] < y[i]







