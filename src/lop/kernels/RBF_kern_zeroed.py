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

# RBF_kern_zeroed.py
# Written Ian Rankin - April 2021
#
# The Radius Basis Function GP kernel. with a modification
# That makes a certian point to always be zero.
# This grounds a preference only GP to have a particular point.

import numpy as np

from lop.kernels import RBF_kern
import pdb

## RBF_kern_zeroed
# Radial basis function for two points.
# This is a single input sample (v (k) and u (k))
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class RBF_kern_zeroed(RBF_kern):

    ## constructor
    # @param sigma - the sigma for the rbf kernel
    # @param l - the lengthscale for the rbf_kernel
    # @param zero_pt - [opt default None [0]] sets the zero point for the RBF_kernel
    # @param sigma_noise [opt default 0.01] sets the amount of noise on sigma
    def __init__(self, sigma, l, zero_pt=None, sigma_noise=0.01):
        super(RBF_kern_zeroed, self).__init__(sigma, l, sigma_noise=sigma_noise)

        self.zero_pt = zero_pt


    def lazy_zero_pt_init(self, u):
        if self.zero_pt is None:
            if isinstance(u, (np.ndarray, list)):
                self.zero_pt = np.zeros(len(u))
            else:
                self.zero_pt = np.array([0])

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
        self.lazy_zero_pt_init(X[0])

       
        cov = super().cov(X, Y)
        cov_zero = self.zero_cov(X, Y)

        print(cov - cov_zero)
        
        
        return cov - cov_zero

    def zero_cov(self, X, Y):
        N = X.shape[0]
        M = Y.shape[0]

        cov_x = super().cov(X, self.zero_pt[np.newaxis,:])
        cov_y = super().cov(self.zero_pt[np.newaxis,:], Y)

        X_expanded = np.repeat(cov_x, M, axis=1)
        Y_expanded = np.repeat(cov_y, N, axis=0)

        pdb.set_trace()

        return X_expanded * Y_expanded


    # get gradient of the covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance gradient tensor of the samples. [n1, n2, k]
    def cov_gradient(self, X, Y):
        self.lazy_zero_pt_init(X[0])

        dSigma, dl = super().cov_gradient(X,Y)

        N = X.shape[0]
        M = Y.shape[0]

        cov_x = super().cov(X, self.zero_pt[np.newaxis,:])
        cov_y = super().cov(self.zero_pt[np.newaxis,:], Y)

        X_expanded = np.repeat(cov_x, M, axis=1)
        Y_expanded = np.repeat(cov_y, N, axis=0)

        x_sig, x_l = super().cov_gradient(X, self.zero_pt[np.newaxis,:])
        y_sig, y_l = super().cov_gradient(self.zero_pt[np.newaxis,:], Y)

        # f(x)g'(x) + f'(x)g(x)
        zero_sigma = x_sig * Y_expanded + X_expanded * y_sig
        zero_l =     x_l * Y_expanded + X_expanded * y_l

        return dSigma-zero_sigma, dl-zero_l


    def gradient(self, u, v):
        self.lazy_zero_pt_init(u)

        grad = super().gradient(u, v)

        grad_zero_u = super().gradient(u, self.zero_pt)
        grad_zero_v = super().gradient(v, self.zero_pt)

        # f(x)g'(x) + f'(x)g(x)
        grad_zero = super()(u, self.zero_pt) * grad_zero_v +  grad_zero_u * super()(v, self.zero_pt)

        return grad - grad_zero

    def __call__(self, u, v):
        self.lazy_zero_pt_init(u)
        
        cov = super().__call__(u,v)
        cov_0 = super().__call__(u, self.zero_pt)*super().__call__(v, self.zero_pt)

        return cov - cov_0
