# test_GP.py
# Written Ian Rankin - December 2023
#
#

import pytest
import lop

import numpy as np

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def test_pref_GP_construction():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 1.0))

    assert gp is not None


def test_pref_GP_training_does_not_crash():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 1.0))

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)
    gp.add(X_train, pairs)

    X = np.array([1.5,1.7,3.2])
    y = gp(X)

    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()


def test_pref_GP_function():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))


    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=False)


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


