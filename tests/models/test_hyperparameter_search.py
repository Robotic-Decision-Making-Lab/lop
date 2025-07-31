# test_hyperparameter_search.py
# Written Ian Rankin - January 2024
#
# A set of tests investigating hyperparameter optimization for preference GP's

import pytest
import numpy as np
import random
import pdb

import lop

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_grad_hyper():
    m = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)

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
    m.find_mode(m.X_train, m.y_train)

    grad = m.grad_likli_f_hyper(m.F, X_train, m.y_train)

    assert len(grad) == len(p)
    assert grad[0] < 0
    assert grad[1] > 0
    assert grad[2] < 0


def test_hyperparameter_search_does_not_crash():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)


    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=True)

    assert gp is not None

@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_hyperparameter_with_no_training_data():
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False, use_hyper_optimization=True)


    X = np.array([0,3.2, 4.5])
    y, var = gp.predict(X)

    #gp.add(X_train, pairs)
    #gp.optimize(optimize_hyperparameter=True)

    assert len(y) == 3
    assert gp is not None

    X_train = np.array([0,1])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0)
    gp.add(X_train, pairs)

    y, var = gp.predict(X)

    assert len(y) == 3
    assert gp is not None


@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_get_hyperparameters_multiple_y_trains():
    m = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)

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
    y_abs = lop.normalize_0_1(f_sin(X_abs))

    m.add(X_abs, y_abs, type='abs')

    p = m.get_hyper()
    assert p[0] == 0.5 and p[3] == 0.5 and p[4] == 0.7

@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_set_hyperparameters_multiple_y_trains():
    m = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)


    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    m.add(X_train, pairs)


    X_abs = np.array([1.5, 2.5])
    y_abs = lop.normalize_0_1(f_sin(X_abs))

    m.add(X_abs, y_abs, type='abs')

    p_set = np.array([0.4,0.3,0.2,0.6,1.2])
    m.set_hyper(p_set)
    p = m.get_hyper()
    assert (p == p_set).all()

@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_hyperparameter_search_somewhat_converges():
    np.random.seed(1)
    random.seed(0)
    
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)


    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=True)


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()

    assert (mu < 20).all()
    assert (mu > -20).all()


    y, sigma = gp.predict(X_train)

    bound = 0.2

    #pdb.set_trace()
    for i in range(len(X_train)):
        if i != 0:
            assert y[0] > y[i] - bound
        if i!= 1:
            assert y[1] < y[i] + bound

@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_hyperparameter_search_converges_only_abs_bound():
    np.random.seed(0)
    random.seed(0)

    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), hyperparam_only_probit=False)


    X_abs = np.array([1.5, 2.5])
    y_abs = lop.normalize_0_1(f_sin(X_abs), 0.05)

    gp.add(X_abs, y_abs, type='abs')

    gp.optimize(optimize_hyperparameter=True)


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()


    X_train = np.array([0,1,2,3,4.2,6,7])
    y, sigma = gp.predict(X_train)

    assert (y < 20).all()
    assert (y > -10).all()

@pytest.mark.skip(reason="Hyperparameter optimization has never worked well and is mostly disabled")
def test_hyperparameter_search_somewhat_converges_abs_bound_pairs():
    np.random.seed(0)
    random.seed(0)
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))


    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    gp.add(X_train, pairs)

    X_abs = np.array([0.2, 1.5, 2.5, 4.7])
    y_abs = lop.normalize_0_1(f_sin(X_abs), 0.05)

    gp.add(X_abs, y_abs, type='abs')

    np.random.seed(0)
    random.seed(0)

    gp.optimize(optimize_hyperparameter=True)


    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    assert not np.isnan(mu).any()
    assert not np.isnan(sigma).any()
    assert not np.isnan(std).any()


    y, sigma = gp.predict(X_train)

    num_satisfied = 0

    for i, pair in enumerate(pairs):
        if y[pair[1]] > y[pair[2]]:
            if pair[0] == lop.get_dk(1,0):
                num_satisfied += 1
        else:
            if pair[0] == lop.get_dk(0,1):
                num_satisfied += 1

    assert (num_satisfied / len(pairs)) > 0.9

    # for i in range(len(X_train)):
    #     if i != 0:
    #         assert y[0] > y[i] - bound
    #     if i!= 1:
    #         assert y[1] < y[i] + bound







