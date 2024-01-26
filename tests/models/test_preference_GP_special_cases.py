# test_preference_GP_special_cases.py
# Written Ian Rankin - December 2023
#
# A set of specefic tests to ensure working optimization and convergence.
# Even in some relatively explicit conditions.

import pytest
import lop

import numpy as np

import pdb


def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def test_prior_only():
    X_train = np.array([0,1,2,3,4.2,6,7])

    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), normalize_gp=False, normalize_positive=False)
    gp.add(X_train, [])
    gp.optimize(optimize_hyperparameter=False)

    # check that the F values are fairly reasonable (should be close to 0)
    assert (np.abs(gp.F) < 0.05).all()

    X = np.array([1.5, 2.4, 6.7, 2.3])
    mu, sigma = gp.predict(X)

    assert (np.abs(mu) < 0.05).all()
    assert (np.abs(sigma - 0.25) < 0.05).all()




def test_close_pts_K_near_singularity():
    # setup training data
    X_train = np.array([0,1.9999,2,2.0001,4.2,6,7, 4.7])
    y_train = f_sin(X_train)

    pairs = lop.gen_pairs_from_idx(np.argmax(y_train[0:3]), list(range(len(y_train[0:3]))))
    
    pairs2 = lop.gen_pairs_from_idx(np.argmax(y_train[3:5]), list(range(len(y_train[3:5]))))
    pairs2 = [(p[0], p[1]+3, p[2]+3) for p in pairs2]
    pairs += pairs2
    pairs2= lop.gen_pairs_from_idx(np.argmax(y_train[5:]), list(range(len(y_train[5:]))))
    pairs2 = [(p[0], p[1]+5, p[2]+5) for p in pairs2]
    pairs += pairs2

    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), normalize_gp=False)
    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=False)

    assert (np.abs(gp.F) < 20).all()

    # predict output of GP, large set of outputs
    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)

    assert (np.abs(mu) < 20).all()

    # predict output of GP small subset
    X = np.array([0.5, 1.5, 3, 4, 5])
    y = f_sin(X)

    mu, sigma = gp.predict(X)

    assert (np.abs(mu-y) < 0.4).all()

