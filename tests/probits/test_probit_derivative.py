# Copyright 2024 Ian Rankin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# test_probits.py
# Written Ian Rankin - Febuary 2024
#
# A set of test functions for derivatives of the probit function to parameters

import pytest
import lop

import numpy as np
import pdb

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def test_preference_probit_calc_W_df():
    pp = lop.PreferenceProbit(2.0)

    X_train = np.array([0,1,2,3,4.2,6,7])
    F = np.array([1,0.5,3,4,5,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

    dW = pp.calc_W_dF(pairs, F)

    assert dW.shape[0] == dW.shape[1]
    assert dW.shape[1] == dW.shape[2]
    assert dW.shape[0] == len(F)

def test_preference_probit_calc_W_dHyper():
    pp = lop.PreferenceProbit(2.0)

    X_train = np.array([0,1,2,3,4.2,6,7])
    F = np.array([1,0.5,3,4,5,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

    dW = pp.calc_W_dHyper(pairs, F)

    assert dW.shape[0] == 1
    assert dW.shape[1] == len(F)
    assert dW.shape[2] == len(F)

    assert not np.isnan(dW).any()


def test_preference_probit_derivative_hyper():
    pp = lop.PreferenceProbit(2.0)

    X_train = np.array([0,1,2,3,4.2,6,7])
    F = np.array([1,0.5,3,4,5,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

    grad = pp.grad_hyper(pairs, F)

    assert len(grad) == 1
    assert not np.isnan(grad[0])


def test_preference_probit_param_likli():
    pp = lop.PreferenceProbit()

    likli = pp.param_likli()

    assert not np.isnan(likli)

    grad_likli = pp.grad_param_likli()

    assert not np.isnan(grad_likli).any()
    assert len(grad_likli) == 1


########### abs_bound tests

def test_abs_bound_probit_calc_W_df():
    pro = lop.AbsBoundProbit()

    #X_train = np.array([0,1,2])
    F = np.array([0.4, 0.3, 0.6])
    v = np.array([0.1,0.2,0.5])
    idxs = np.arange(0,3,1)

    dW = pro.calc_W_dF((v, idxs), F)

    assert dW.shape[0] == dW.shape[1]
    assert dW.shape[1] == dW.shape[2]
    assert dW.shape[0] == len(F)

    assert not np.isnan(dW).any()



def test_abs_bound_probit_calc_W_dHyper():
    pro = lop.AbsBoundProbit()

    #X_train = np.array([0,1,2])
    F = np.array([0.4, 0.3, 0.6])
    v = np.array([0.1,0.2,0.5])
    idxs = np.arange(0,3,1)

    dW = pro.calc_W_dHyper((v, idxs), F)

    assert dW.shape[0] == 2
    assert dW.shape[1] == len(F)
    assert dW.shape[2] == len(F)

    assert not np.isnan(dW).any()

def test_abs_bound_probit_derivative_hyper():
    pro = lop.AbsBoundProbit()

    #X_train = np.array([0,1,2])
    F = np.array([0.4, 0.3, 0.6])
    v = np.array([0.1,0.2,0.5])
    idxs = np.arange(0,3,1)

    grad = pro.grad_hyper((v, idxs), F)

    assert len(grad) == 2
    assert not np.isnan(grad).all()

def test_abs_bound_param_likli():
    pp = lop.AbsBoundProbit()

    likli = pp.param_likli()

    assert not np.isnan(likli)


    grad_likli = pp.grad_param_likli()

    assert not np.isnan(grad_likli).any()
    assert len(grad_likli) == 2


@pytest.mark.skip()
def test_ordinal_probit_calc_W_df():
    assert False

@pytest.mark.skip()
def test_ordinal_probit_calc_W_dHyper():
    assert False

@pytest.mark.skip()
def test_ordinal_probit_derivative_hyper():
    assert False

@pytest.mark.skip()
def test_ordinal_probit_param_likli():
    pp = lop.OrdinalProbit()

    likli = pp.param_likli()

    assert not np.isnan(likli).any()
    assert len(likli) == 2

    grad_likli = pp.grad_param_likli()

    assert not np.isnan(grad_likli).any()
    assert len(grad_likli) == 2
