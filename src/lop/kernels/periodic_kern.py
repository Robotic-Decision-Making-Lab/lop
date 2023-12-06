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

# periodic_kern.py
# Written Ian Rankin - September 2021
#
# The periodic kernel Function for GPs.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from lop.kernels import kernel_func


## periodic_kern
# A periodic kernel for gaussian processes
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class periodic_kern(kernel_func):

    ## Constructor
    # @param sigma - the sigma for the rbf kernel
    # @param - sigma, the sigma for the periodic kernel
    # @param - l, the lengthscale for the periodic kernel
    # @param - p, the periodicity of the periodic kernel
    def __init__(self, sigma, l, p):
        super(periodic_kern, self).__init__()

        self.sigma = sigma
        self.l = l
        self.p = p


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
        diff = X_expanded - Y_expanded
        uv_norm = np.sum(np.abs(diff), axis=2)
        
        sin_tmp = np.sin(np.pi*uv_norm / self.p)
        exp_tmp = -2 * sin_tmp * sin_tmp / (self.l * self.l)

        cov = self.sigma * self.sigma * np.exp(exp_tmp)
        return cov


    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.l = theta[1]
        self.p = theta[2]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.l, self.p])
        return theta

    def gradient(self, u, v):
        uv_norm = np.sum(np.abs(u-v)) #np.linalg.norm(u-v, ord=1)
        sin_tmp = np.sin(np.pi*uv_norm / self.p)
        exp_int = - 2 * sin_tmp * sin_tmp / (self.l*self.l)
        exp_x = np.exp(exp_int)

        dSigma = 2 * self.sigma * exp_x

        dl = -2 * self.sigma*self.sigma * exp_x * uv_norm / (self.l*self.l*self.l)

        dp = 2 * np.pi * self.sigma * self.sigma * uv_norm * exp_x * \
            np.cos(np.pi * uv_norm / self.p) / (self.l*self.l * self.p*self.p)

        return np.array([dSigma, dl, dp])

    def __call__(self, u, v):
        uv_norm = np.sum(np.abs(u-v)) #np.linalg.norm(u-v, ord=1)
        sin_tmp = np.sin(np.pi*uv_norm / self.p)
        exp_int = - 2 * sin_tmp * sin_tmp / (self.l*self.l)

        return self.sigma*self.sigma * np.exp(exp_int)

    def __len__(self):
        return 3
