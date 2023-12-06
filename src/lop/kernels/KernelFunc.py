# Copyright 2021 Ian Rankin
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

# KernelFunc.py
# Written Ian Rankin - September 2021
#
# The base class for all kernel functions in lop.
# Additionally adds the DualKern function to handle adding or 
# multiplying kernel functions.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence



## Base kernel function class
class KernelFunc:
    def __init__(self):
        pass

    ## get covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance matrix of the samples.
    def cov(self, X, Y):
        cov = np.empty((len(X), len(Y)))

        for i,x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                cov_ij = self.__call__(x1, x2)
                cov[i,j] = cov_ij
        return cov

    ## get gradient of the covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance gradient tensor of the samples. [n1, n2, k]
    def cov_gradient(self, X, Y):
        cov = np.empty((len(X), len(Y), len(self)))

        for i,x1 in enumerate(X):
            for j,x2 in enumerate(Y):
                cov_ij = self.gradient(x1, x2)
                cov[i,j, :] = cov_ij
        return cov

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        raise NotImplementedError('KernelFunc update function not implemented')

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        return np.empty(0)

    def gradient(self, u, v):
        raise NotImplementedError('KernelFunc gradient function not implemented')

    def __call__(self, u, v):
        raise NotImplementedError('KernelFunc __call__ function not implemented')

    # self + other
    def __add__(self, other):
        if isinstance(other, KernelFunc):
            return DualKern(self, other, '+')
        else:
            raise TypeError('kernel function add passed a type ' + str(type(other)))

    # self * other
    def __mul__(self, other):
        if isinstance(other, KernelFunc):
            return DualKern(self, other, '*')
        else:
            raise TypeError('kernel function multiply passed a type ' + str(type(other)))

    def __len__(self):
        return 0


## kernel function class to handle adding or multiplying
# kernel functions together.
# allows stacking of kernel function such as.
# (gr.RBF_kern(1,1) * gr.periodic_kern(1,1,1)) + gr.linear_kern(1,1,1)
class DualKern(KernelFunc):
    ## Constructor
    # @param kern_1 - the first KernelFunc
    # @param kern_2 - the second KernelFunc
    # @param operator - the operator to apply between the two kernel function
    #           supports ['+', '*']
    def __init__(self, kern_1, kern_2, operator='+'):
        super(DualKern, self).__init__()
        self.a = kern_1
        self.b = kern_2

        self.operator = operator

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        if isinstance(theta, Sequence) and not isinstance(theta, np.ndarray):
            theta = np.array(theta)

        self.a.set_param(theta[:len(self.a)])
        self.b.set_param(theta[len(self.a):])


    ## get covariance matrix
    # calculate the covariance matrix between the samples given in X
    # overiding the KernelFunc get covariance matrix in order to vectorize
    # and increase the speed of computation.
    # @param X - samples (n1,k) numpy array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k) numpy array
    #
    # @return the covariance matrix of the samples.
    def cov(self, X, Y):
        a_f = self.a.cov(X,Y)
        b_f = self.b.cov(X,Y)

        if self.operator == '+':
            return a_f + b_f
        elif self.operator == '*':
            return a_f * b_f
        else:
            raise NotImplementedError('DualKern does not have operator `'+self.operator+'` implemented')

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        a_param = self.a.get_param()
        b_param = self.b.get_param()

        return np.append(a_param, b_param, axis=0)

    def gradient(self, u, v):
        if self.operator == '+':
            a_grad = self.a.gradient(u,v)
            b_grad = self.b.gradient(u,v)
        elif self.operator == '*':
            a_grad = self.a.gradient(u,v) * self.b(u,v)
            b_grad = self.b.gradient(u,v) * self.a(u,v)
        else:
            raise NotImplementedError('DualKern does not have operator `'+self.operator+'` implemented')

        return np.append(a_grad, b_grad, axis=0)

    def __call__(self, u, v):
        a_f = self.a(u,v)
        b_f = self.b(u,v)

        if self.operator == '+':
            return a_f + b_f
        elif self.operator == '*':
            return a_f * b_f
        else:
            raise NotImplementedError('DualKern does not have operator `'+self.operator+'` implemented')

    def __len__(self):
        return len(self.a)+len(self.b)

