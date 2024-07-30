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





# test the mean link can handle both and 1d and 2d inputs
def test_abs_bound_1d_get_mean_link():
    probit = lop.AbsBoundProbit()

    F = np.array([0.5, 0.7, 1.2, 0.9])

    ml = probit.mean_link(F)

    assert ml.shape[0] == 4
    assert len(ml.shape) == 1

# test the cdf function output reasonable outputs
def test_abs_bound_2d_get_mean_link():
    probit = lop.AbsBoundProbit()

    F = np.array([[0.5, 0.7], [1.2, 0.9]])

    ml = probit.mean_link(F)

    assert ml.shape[0] == 2
    assert ml.shape[1] == 2
    assert len(ml.shape) == 2

def test_get_alpha_beta_1d():
    probit = lop.AbsBoundProbit()

    F = np.array([0.5, 0.7, 1.2, 0.9])

    aa, bb = probit.get_alpha_beta(F)

    assert aa.shape == bb.shape
    assert len(aa.shape) == 1
    assert aa.shape[0] == 4

def test_get_alpha_beta_2d():
    probit = lop.AbsBoundProbit()

    F = np.array([[0.5, 0.7], [1.2, 0.9]])

    aa, bb = probit.get_alpha_beta(F)

    assert aa.shape == bb.shape
    assert len(aa.shape) == 2
    assert aa.shape[0] == 2
    assert aa.shape[1] == 2