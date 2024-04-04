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

# test_GP_kernel.py
# Written Ian Rankin - October 2022 (Based on tests written oct. 2021)
#
# 

import pytest

import lop
import numpy as np

import pdb




def test_vectorization_rbf():
    k = lop.RBF_kern_zeroed(1,0.8)

    X = np.array([1,3,4,5,6,7])
    Y = np.array([-1,-0.5,0,1,2,3])


    cov = np.empty((len(X), len(Y)))

    for i,x1 in enumerate(X):
        for j,x2 in enumerate(Y):
            cov_ij = k(x1, x2)
            cov[i,j] = cov_ij


    cov_vec = k.cov(X,Y)

    for i in range(cov_vec.shape[0]):
        for j in range(cov_vec.shape[1]):
            assert cov[i,j] == cov_vec[i,j]


def test_rbf():
    rbf = lop.RBF_kern_zeroed(1, 1)
    assert rbf(1,2) > 0.5 # probably right
    assert rbf(1,2) < 0.7 # probably right

def test_rbf_grad_cov():
    rbf = lop.RBF_kern_zeroed(1, 1)

    X = np.array([0,1,3,4,5,6,7])

    dK_sigma, dK_l = rbf.cov_gradient(X,X)

    assert dK_sigma.shape[0] == len(X) and dK_sigma.shape[1] == len(X)
    assert dK_l.shape[0] == len(X) and dK_l.shape[1] == len(X)
    assert dK_l[0,0] == 0


def test_rbf_param_liklihood():
    rbf = lop.RBF_kern_zeroed(1, 1)

    liklihood = rbf.param_likli()
    assert not np.isnan(liklihood)
    assert liklihood < 10.0

def test_rbf_grad_param_liklihood():
    rbf = lop.RBF_kern_zeroed(1, 1)

    X = np.array([1,3,4,5,6,7])

    d_liklihood = rbf.grad_param_likli()
    assert not np.isnan(d_liklihood).any()
    assert len(d_liklihood) == 2

def test_rbf_kern_cov():
    rbf_test = lop.RBF_kern_zeroed(1, 1)
    X = np.array([0,1,2,3,4.2,6,7])


    c = rbf_test.cov(X,X)

    assert np.abs(c[0,0]) < 0.015
    
    assert c[-1,0] < 0.0001
    assert c[0,-1] < 0.0001

    assert np.linalg.det(c) > 0