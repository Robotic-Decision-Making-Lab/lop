# test_preference_GP.py
# Written Ian Rankin - December 2023
#
# A set of relatively generic tests on the GP. To ensure some 
# amount of convergence and working predictions

import pytest
import lop

import numpy as np

import pdb

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def f_sq(x, data=None):
    return (x/10.0)**2

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

def test_pref_gp_predict_without_training():
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))

    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()

def test_pref_gp_predict_large_without_training():
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))

    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict_large(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()

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




def test_pref_GP_abs_bound():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 0.7), normalize_positive=True)

    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sq(X_train)

    gp.add(X_train, y_train, type='abs')
    
    assert gp is not None

    gp.optimize()

    assert gp is not None
    assert gp.optimized
    assert gp.n_loops > 0 and gp.n_loops < 90

    X = np.array([1.5,1.7,3.2])
    y = gp(X)

    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()


def test_pref_GP_abs_bound_single_rate():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 0.7), pareto_pairs=True)

    X_train = np.array([0.2])
    y_train = f_sq(X_train)

    #pdb.set_trace()

    gp.add(X_train, y_train, type='abs')

    assert gp is not None

    gp.optimize()

    assert gp is not None
    assert gp.optimized
    assert gp.n_loops > 0 and gp.n_loops < 90

    X = np.array([1.5,1.7,3.2])
    y = gp(X)

    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()

def test_pref_GP_pref_abs_does_not_crash():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 0.7), normalize_positive=True)

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sq, 0) + \
            lop.generate_fake_pairs(X_train, f_sq, 1) + \
            lop.generate_fake_pairs(X_train, f_sq, 2) + \
            lop.generate_fake_pairs(X_train, f_sq, 3) + \
            lop.generate_fake_pairs(X_train, f_sq, 4)

   
    gp.add(X_train, pairs)

    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sq(X_train)

    gp.add(X_train, y_train, type='abs')
    
    assert gp is not None

    gp.optimize()

    assert gp is not None
    assert gp.optimized
    assert gp.n_loops > 0 and gp.n_loops < 90

    X = np.array([1.5,1.7,3.2])
    y = gp(X)

    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()


def test_pref_GP_pref_abs_does_not_crash():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 0.7), normalize_positive=True)

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sq, 0) + \
            lop.generate_fake_pairs(X_train, f_sq, 1) + \
            lop.generate_fake_pairs(X_train, f_sq, 2) + \
            lop.generate_fake_pairs(X_train, f_sq, 3) + \
            lop.generate_fake_pairs(X_train, f_sq, 4)

   
    gp.add(X_train, pairs)

    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sq(X_train)

    gp.add(X_train, y_train, type='abs')
    
    assert gp is not None

    gp.optimize()

    assert gp is not None
    assert gp.optimized
    assert gp.n_loops > 0 and gp.n_loops < 90

    X = np.array([1.5,1.7,3.2])
    y = gp(X)

    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()

def f_sin_ord(x, data=None):
    return 0.5 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x)) + 0.5

def test_pref_GP_pref_ordinal_does_not_crash():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin_ord, 0) + \
            lop.generate_fake_pairs(X_train, f_sin_ord, 1) + \
            lop.generate_fake_pairs(X_train, f_sin_ord, 2)

    # Abs bound ordinal values
    X_train_ord = np.array([0.1, 4.7, 6.3])
    y_train_ord = (f_sin_ord(X_train_ord)*5).astype(int)


    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))
    gp.set_num_ordinals(5)
    gp.add(X_train, pairs)
    gp.add(X_train_ord, y_train_ord, type='ordinal')
    gp.optimize(optimize_hyperparameter=False)

    assert gp is not None
   
    gp.add(X_train, pairs)

    X = np.array([1.5,1.7,3.2])
    y = gp(X)


    assert isinstance(y, np.ndarray)
    assert not np.isnan(y).any()


