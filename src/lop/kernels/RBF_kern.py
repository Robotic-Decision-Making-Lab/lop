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

# RBF_kern.py
# Written Ian Rankin - September 2021
#
# The Radius Basis Function GP kernel.

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from lop.kernels import KernelFunc
from lop.utilities import log_pdf_gamma, d_log_pdf_gamma

## RBF_kern
# Radial basis function for two points.
# This is a single input sample (v (k) and u (k))
# @param u - a single input sample
# @param v - a second input sample of the same dimension
class RBF_kern(KernelFunc):

    ## Constructor
    # @param sigma - the sigma for the rbf kernel
    # @param l - the lengthscale for the rbf_kernel
    def __init__(self, sigma, l, sigma_noise=0.01):
        super(RBF_kern, self).__init__()

        self.sigma = sigma
        self.l = l
        self.sigma_noise = sigma_noise

        # prior on hyper-parameters
        self.sigma_k = 10
        self.sigma_theta = 0.1
        self.l_k = 3.0
        self.l_theta = 0.4

    # update the parameters
    # @param theta - vector of parameters to update
    def set_param(self, theta):
        self.sigma = theta[0]
        self.l = theta[1]
        #self.sigma_noise = theta[2]

    # get_param
    # get a vector of the parameters for the kernel function (used for hyper-parameter optimization)
    def get_param(self):
        theta = np.array([self.sigma, self.l])
        return theta

    ## param_likli
    # log liklihood of the parameter (prior)
    # for RBF kernels this is a parameterized gamma_distribution. Scaled for functions of 
    # approximently size 1 and distance between points in [0,10] ish range
    def param_likli(self):
        return log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta) + \
                log_pdf_gamma(self.l, self.l_k, self.l_theta)

    ## grad_param_likli
    # gradient of the log liklihood of the parameter (prior)
    # @return numpy array of gradient of each parameter
    def grad_param_likli(self):
        return np.array([d_log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta),
                d_log_pdf_gamma(self.l, self.l_k, self.l_theta)])

    
    ## Performs random sampling using the same liklihood function used by the param
    # liklihood function
    # @return numpy array of independent samples.
    def randomize_hyper(self):
        return np.array([
            np.random.gamma(self.sigma_k, self.sigma_theta),
            np.random.gamma(self.l_k, self.l_theta)])

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
        top = np.sum(diff*diff, axis=2)

        cov = self.sigma * self.sigma * np.exp(-top / (2 * self.l*self.l))
        if cov.shape[0] == cov.shape[1] and (X[0] == Y[0]).all():
            cov += np.eye(cov.shape[0])*self.sigma_noise
        return cov

    # get gradient of the covariance matrix
    # calculate the covariance matrix between the samples given in X
    # @param X - samples (n1,k) array where n is the number of samples,
    #        and k is the dimension of the samples
    # @param Y - samples (n2, k)
    #
    # @return the covariance gradient tensor of the samples. [n1, n2, k]
    def cov_gradient(self, X, Y):
        N = X.shape[0]
        M = Y.shape[0]

        if len(X.shape) == 1:
            X = X[:,np.newaxis]
        if len(Y.shape) == 1:
            Y = Y[:,np.newaxis]


        X_expanded = np.repeat(X[:,np.newaxis,:], M, axis=1)
        Y_expanded = np.repeat(Y[np.newaxis,:,:], N, axis=0)
        diff = X_expanded - Y_expanded
        top = np.sum(diff*diff, axis=2)

        exp_x = np.exp(-top / (2 * self.l*self.l))

        dSigma = 2 * self.sigma * exp_x
        dl = 2 * self.sigma * self.sigma * top * exp_x / (self.l*self.l*self.l)
        # if N == N:
        #     dSigma_noise = np.eye(N)
        # else:
        #     dSigma_noise = np.zeros((N, M))

        return dSigma, dl


    def gradient(self, u, v):
        top = (u-v)
        top = np.sum(top*top)

        exp_x = np.exp(-top / (2 * self.l*self.l))

        dSigma = 2 * self.sigma * exp_x
        dl = 2 * self.sigma * self.sigma * top * exp_x / (self.l*self.l*self.l)

        return np.array([dSigma, dl])

    def __call__(self, u, v):
        top = (u - v)
        top = np.sum(top * top)

        cov = self.sigma*self.sigma * np.exp(-top / (2 * self.l*self.l))# + np.eye(top.shape[0])*self.sigma_noise
        #if (isinstance(u, np.ndarray) and (u == v).all()) or u == v:
        #    cov += self.sigma_noise
        return cov

    def __len__(self):
        return 2