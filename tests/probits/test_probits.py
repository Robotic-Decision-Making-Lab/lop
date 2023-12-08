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


def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_preference_probit():
    pp = lop.PreferenceProbit(2.0)

    X_train = np.array([0,1,2,3,4.2,6,7])
    F = np.array([1,0.5,3,4,5,6,7,2])
    F = F / np.linalg.norm(F, ord=np.inf)
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    pairs = np.array(pairs) # force pairs to be a numpy array for vectorization

    z = pp.z_k(pairs, F)

    W, dpy_df, py = pp.derivatives(pairs, F)

    for z_i in z:
        assert not np.isnan(z_i)

    for py_i in dpy_df:
        assert not np.isnan(py_i)
    assert not np.isnan(py)


