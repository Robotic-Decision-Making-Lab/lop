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

# LinearKern.py
# Written Ian Rankin - September 2021
#
# The linear kernel Function for GPs.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from lop.kernels import KernelFunc



## LinearKern
# A linear kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class LinearKern(KernelFunc):

    ## Constructor
    # @param sigma - the sigma for the linear kernel
    # @param - sigma, the sigma for the linear kernel
    # @param - sigma_b, the sigma_b for the linear
    # @param - c, the offset for the linear kernel
    def __init__(self, sigma, sigma_b, c):
        super(LinearKern, self).__init__()

        self.sigma = sigma
        self.sigma_b = sigma_b
        self.c = c

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.sigma_b = theta[1]
        self.c = theta[2]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.sigma_b, self.c])
        return theta

    ## get covariance matrix
    # calculate the covariance matrix between the samples given in X
    # overiding the kernel_func get covariance matrix in order to vectorize
    # and increase the speed of computation.
    # @param X - samples (n1,k) numpy array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k) numpy array
    #
    # @return the covariance matrix of the samples.
    def cov(self, X, Y):
        N = X.shape[0]
        M = Y.shape[0]

        if len(X.shape) == 1:
            X = X[:,np.newaxis]
        if len(Y.shape) == 1:
            Y = Y[:,np.newaxis]

        X_expanded = np.repeat(X[:,np.newaxis,:], M, axis=1)
        Y_expanded = np.repeat(Y[np.newaxis,:,:], N, axis=0)
        
        tmp = np.sum((X_expanded-self.c) * ( Y_expanded-self.c), axis=2)

        cov = (self.sigma_b**2) + ((self.sigma**2) * tmp)
        return cov


    def gradient(self, u, v):
        dSigma_b = 2 * self.sigma_b

        dSigma = np.sum(2*self.sigma*(u-self.c)*(v-self.c))
        dc = self.sigma*self.sigma*np.sum(2*self.c - u - v)

        return np.array([dSigma, dSigma_b, dc])

    def __call__(self, u, v):
        return (self.sigma_b**2) + ((self.sigma**2) * np.sum((u-self.c) * (v-self.c)))

    def __len__(self):
        return 3

