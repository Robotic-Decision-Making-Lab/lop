# Copyright 2022 Ian Rankin
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
# Written Ian Rankin - December 2023
#
# A set of test functions for each of the probit functions

import pytest
import lop

import numpy as np
import pdb

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

## Preference probit checks

def test_preference_probit_likelihood():
    pp = lop.PreferenceProbit(2.0)

    X_train = np.array([0,1,2,3,4.2,6,7])
    F = np.array([1,0.5,3,4,5,6,7])
    F = F / np.linalg.norm(F, ord=np.inf)
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

    z = pp.z_k(pairs, F)

    W, dpy_df, py = pp.derivatives(pairs, F)

    assert not np.isnan(W).any()
    assert np.isfinite(W).all()
    assert not np.isnan(dpy_df).any()
    assert np.isfinite(dpy_df).all()
    assert not np.isnan(py)
    assert np.isfinite(py)

    assert dpy_df.shape[0] == F.shape[0]
    assert W.shape[0] == F.shape[0]
    assert W.shape[1] == F.shape[0]


def test_preference_hyper_modification():
    probit = lop.PreferenceProbit()

    param = probit.get_hyper()

    assert param.shape[0] == 1

    probit.set_hyper(np.array([0.1]))
    assert probit.sigma == 0.1
    param = probit.get_hyper()

    assert param[0] == 0.1

#### Abs bound checks

def test_abs_bound_probit_likelihood():
    probit = lop.AbsBoundProbit(0.5, 4.0)

    X_train = np.array([0,1,2,3])
    v = np.array([0.1,0.2,0.5, 0.4])
    idxs = np.arange(0,4,1)

    W, dpy_df, py = probit.derivatives((v, idxs), X_train)

    py2 = probit.likelihood((v, idxs), X_train)

    assert not np.isnan(py2)
    assert np.isfinite (py2)

    assert (np.log(py2) - py) < 0.001
    assert (np.log(py2) - py) > -0.001

    assert not np.isnan(W).any()
    assert np.isfinite(W).all()
    assert not np.isnan(dpy_df).any()
    assert np.isfinite(dpy_df).all()
    assert not np.isnan(py)
    assert np.isfinite(py)

    assert dpy_df.shape[0] == X_train.shape[0]
    assert W.shape[0] == X_train.shape[0]
    assert W.shape[1] == X_train.shape[0]

def test_abs_bound_probit_likelihood_not_all_indicies():
    probit = lop.AbsBoundProbit(0.5, 4.0)

    X_train = np.array([0,1,2,3])
    v = np.array([0.1,0.2,0.5])
    idxs = np.arange(0,3,1)

    W, dpy_df, py = probit.derivatives((v, idxs), X_train)

    py2 = probit.likelihood((v, idxs), X_train)

    assert not np.isnan(py2)
    assert np.isfinite (py2)

    assert (np.log(py2) - py) < 0.001
    assert (np.log(py2) - py) > -0.001

    assert not np.isnan(W).any()
    assert np.isfinite(W).all()
    assert not np.isnan(dpy_df).any()
    assert np.isfinite(dpy_df).all()
    assert not np.isnan(py)
    assert np.isfinite(py)

    assert dpy_df.shape[0] == X_train.shape[0]
    assert W.shape[0] == X_train.shape[0]
    assert W.shape[1] == X_train.shape[0]


def test_abs_bound_hyper_modification():
    probit = lop.AbsBoundProbit()

    param = probit.get_hyper()

    assert param.shape[0] == 2

    probit.set_hyper(np.array([0.1, 4.23]))
    assert probit.sigma == 0.1
    assert probit.v == 4.23
    param = probit.get_hyper()

    assert param[0] == 0.1
    assert param[1] == 4.23


# test the cdf function output reasonable outputs
def test_abs_bound_beta_cdf():
    probit = lop.AbsBoundProbit()

    X_train = np.array([0,1,2])
    v = np.array([0.1,0.2,0.5])
    idxs = np.arange(0,3,1)

    cdf = probit.cdf(v, X_train)

    assert not np.isnan(cdf).any()
    assert (cdf > 0.0).all()
    assert (cdf < 1.0).all()



### Ordinal probit checks

def test_ordinal_probit_likelihood():
    probit = lop.OrdinalProbit()

    X_train = np.array([0,1.0,2,3])
    v = np.array([2,1,1,3], dtype=int)
    idxs = np.arange(0,4,1)

    W, dpy_df, py = probit.derivatives((v, idxs), X_train)

    py2 = probit.likelihood((v, idxs), X_train)

    assert not np.isnan(py2)
    assert np.isfinite (py2)

    assert (np.log(py2) - py) < 0.001
    assert (np.log(py2) - py) > -0.001


    assert not np.isnan(W).any()
    assert np.isfinite(W).all()
    assert not np.isnan(dpy_df).any()
    assert np.isfinite(dpy_df).all()
    assert not np.isnan(py)
    assert np.isfinite(py)

    assert dpy_df.shape[0] == X_train.shape[0]
    assert W.shape[0] == X_train.shape[0]
    assert W.shape[1] == X_train.shape[0]

def test_ordinal_not_all_indicies_in_training():
    probit = lop.OrdinalProbit()

    X_train = np.array([0,1.0,2,3])
    v = np.array([2,1,1], dtype=int)
    idxs = np.arange(0,3,1)

    W, dpy_df, py = probit.derivatives((v, idxs), X_train)

    py2 = probit.likelihood((v, idxs), X_train)

    assert not np.isnan(py2)
    assert np.isfinite (py2)

    assert (np.log(py2) - py) < 0.001
    assert (np.log(py2) - py) > -0.001


    assert not np.isnan(W).any()
    assert np.isfinite(W).all()
    assert not np.isnan(dpy_df).any()
    assert np.isfinite(dpy_df).all()
    assert not np.isnan(py)
    assert np.isfinite(py)

    assert dpy_df.shape[0] == X_train.shape[0]
    assert W.shape[0] == X_train.shape[0]
    assert W.shape[1] == X_train.shape[0]



def test_ordinal_hyper_modification():
    probit = lop.OrdinalProbit()

    param = probit.get_hyper()

    assert param.shape[0] == 2

    probit.set_hyper(np.array([0.1, 4.23]))
    assert probit.sigma == 0.1
    assert probit.b[-2] == 4.23
    param = probit.get_hyper()

    assert param[0] == 0.1
    assert param[1] == 4.23
