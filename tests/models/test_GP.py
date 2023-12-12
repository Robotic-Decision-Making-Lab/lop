# test_model.py
# Written Ian Rankin - December 2023
#
#

import pytest
import lop

import numpy as np

def test_GP_construction():
    gp = lop.GP(lop.RBF_kern(1.0, 1.0))

    assert gp is not None




# Test the basic GP is working properly with a couple basic tests.
def test_simple_GP_prediction():
    X_train = np.array([0,1,2,3,6,7])
    X = np.arange(-3, 12, 0.1)
    y_train = np.array([1, 0.5,0, -1, 1, 2])
    training_sigma=np.array([1, 0.5, 0.1, 0.1, 0.2, 0])

    gp = lop.GP(lop.RBF_kern(1,1)+lop.PeriodicKern(1,1,10)+lop.LinearKern(3,1,0.3))
    gp.add(X_train, y_train, training_sigma=training_sigma)

    mu, sigma = gp.predict(X)

    pre = 0.5
    assert mu[30] < y_train[0]+pre
    assert mu[30] > y_train[0]-pre

    assert mu[40] < y_train[1]+pre
    assert mu[40] > y_train[1]-pre

    mu = gp(X)

    pre = 0.5
    assert mu[30] < y_train[0]+pre
    assert mu[30] > y_train[0]-pre

    assert mu[40] < y_train[1]+pre
    assert mu[40] > y_train[1]-pre


def test_gp_reset():
    gp = lop.GP(lop.RBF_kern(1.0, 1.0))
    X = np.arange(-3, 12, 0.1)
    X_train = np.array([0,1,2,3,6,7])
    y_train = np.array([1, 0.5,0, -1, 1, 2])

    gp.add(X_train, y_train)
    mu = gp(X)

    assert not np.isnan(mu).any()

    gp.reset()
    assert gp is not None

    mu = gp(X)
    assert not np.isnan(mu).any()
    assert (mu == 0).all()


def f_sin(x, data=None):
    x = 6-x
    return 2 * np.cos(np.pi * (x[:,0]-2)) * np.exp(-(0.9*x[:,0])) + 1.5 * np.cos(np.pi * (x[:,1]-2)) * np.exp(-(1.2*x[:,1]))

def f_lin(x, data=None):
    return x[:,0]*x[:,1]


