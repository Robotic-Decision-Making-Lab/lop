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
import sys

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

def test_get_alpha_beta_3d():
    probit = lop.AbsBoundProbit()

    F = np.array([[[0.5], [0.7]], [[1.2], [0.9]]])

    aa, bb = probit.get_alpha_beta(F)

    assert aa.shape == bb.shape
    assert len(aa.shape) == 3
    assert aa.shape[0] == 2
    assert aa.shape[1] == 2
    assert aa.shape[2] == 1

@pytest.mark.skipif('numba' not in sys.modules, reason='requires the numba library')
def test_numba_beta_pdf():
    res1 = lop.numba_beta_pdf(0.5, 0.5, 0.5)

    assert res1 > 0.5 and res1 < 0.8

    res2 = lop.numba_beta_pdf(np.array([0.1,0.2,0.3,0.7,0.8]), 0.5, 0.5)

    assert res2.shape[0] == 5
    assert (res2 > 0).all()


    aa = np.array([3,1,6,7,8])
    bb = np.array([1,7,2,4,9])
    res3 = lop.numba_beta_pdf1(np.array([0.1,0.2,0.3,0.7,0.8]), aa, bb)

    assert res3.shape[0] == 5
    assert (res3 > 0).all()

    aa = np.array([[1,2], [3,4]])
    bb = np.array([[2,5], [6,4]])
    res3 = lop.numba_beta_pdf2(np.array([[0.1, 0.2], [0.3, 0.4]]), aa, bb)

    assert res3.shape[0] == 2 and res3.shape[1] == 2
    assert (res3 > 0).all()

    aa = np.array([[[1],[2]], [[3],[4]]])
    bb = np.array([[[2],[5]], [[6],[4]]])
    res4 = lop.numba_beta_pdf3(np.array([[[0.1], [0.2]], [[0.3], [0.4]]]), aa, bb)

    assert len(res4.shape) == 3
    assert res4.shape[0] == 2 and res4.shape[1] == 2
    assert (res4 > 0).all()

    aa = np.array([[[1],[2]], [[3],[4]]])
    bb = np.array([[[2],[5]], [[6],[4]]])
    res5 = lop.numba_beta_pdf3(0.3, aa, bb)

    assert len(res5.shape) == 3
    assert res5.shape[0] == 2 and res5.shape[1] == 2
    assert (res5 > 0).all()


@pytest.mark.skipif('numba' not in sys.modules, reason='requires the numba library')
def test_numba_beta_pdf():
    res1 = lop.numba_beta_pdf(0.5, 200, 200)

    assert not np.isnan(res1)
    