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




def test_vectorization_dual():
    k = lop.RBF_kern(1,0.8) + (lop.PeriodicKern(1,0.8, 10) * lop.LinearKern(1,0.8, 0.5))

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


def test_combined_kern():
    rbf = lop.RBF_kern(1, 1)
    kern2 = lop.PeriodicKern(1,1, 3)
    kern3 = lop.LinearKern(1,1,1)

    combined = rbf + (kern2 * kern3)

    assert type(combined) is lop.DualKern
    assert combined(1,2) > 0.8 # probably right
    assert combined(1,2) < 0.9 # probably right

    param_l = [1,1,1,1,3,1,1,1]
    for i in range(len(param_l)):
        assert combined.get_param()[i] == param_l[i]

    combined.gradient(1,2) # just check this is running not gonna check values

    param_l = [2,3,5,4,7, 1,1,2]
    combined.set_param(param_l)
    for i in range(len(param_l)):
        assert combined.get_param()[i] == param_l[i]


def test_dual_grad_cov():
    kern = lop.LinearKern(1, 1,1) + lop.RBF_kern(1,1)

    X = np.array([1,3,4,5,6,7])

    dK_all = kern.cov_gradient(X,X)


    for dK_i in dK_all:
        assert dK_i.shape[0] == len(X) and dK_i.shape[1] == len(X)

def test_combined_grad_cov():
    kern = lop.LinearKern(1, 1,1) + lop.RBF_kern(1,1) + lop.PeriodicKern(1,1,3)

    X = np.array([1,3,4,5,6,7])

    dK_all = kern.cov_gradient(X,X)

    for dK_i in dK_all:
        assert dK_i.shape[0] == len(X) and dK_i.shape[1] == len(X)


def test_combined_param_liklihood():
    rbf = lop.LinearKern(1, 1,1) + lop.RBF_kern(1,1) + lop.PeriodicKern(1,1,3)

    liklihood = rbf.param_likli()
    assert not np.isnan(liklihood)
    assert liklihood < 10.0

def test_combined_grad_param_liklihood():
    rbf = lop.LinearKern(1, 1,1) + lop.RBF_kern(1,1) + lop.PeriodicKern(1,1,3)

    X = np.array([1,3,4,5,6,7])

    d_liklihood = rbf.grad_param_likli()
    assert not np.isnan(d_liklihood).any()
    assert len(d_liklihood) == 8


