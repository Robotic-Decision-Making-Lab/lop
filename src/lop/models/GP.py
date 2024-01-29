# Copyright 2023 Ian Rankin
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

# GP.py
# Written Ian Rankin - December 2023
#
# A base Gaussian process implementation.
# Useful for testing out various active learning algorithms and ensuring code is working.

import numpy as np
from scipy.linalg import cho_solve
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from lop.models import Model

class GP(Model):

    ## Constructor
    # @param cov_func - the covariance function to use
    # @param mat_inv - [opt] the matrix inversion function to use. By default
    #                   just uses numpy.linalg.inv
    # @param mean_func - [opt] a function that modifies the normal 0 mean GP
    #                   this simply adds the GP estimate to the given function.
    #                   must be able to take vectorized inputs.
    def __init__(self, cov_func, mat_inv=np.linalg.pinv, mean_func=None, active_learner=None):
        super(GP, self).__init__(active_learner)
        self.cov_func = cov_func

        self.invert_function = mat_inv
        self.X_train = None
        self.y_train = None

        if mean_func is None:
            self.mean_func = lambda x : 0
        else:
            self.mean_func = mean_func

    ## add
    # adds training data to the gaussian process
    # appends the data if there already is some training data
    # @param X - the input training data
    # @param y - the output labels of the training data
    # @param training_sigma - [opt] sets the uncertianty in the training data
    #                          accepts scalars or a vector if each sample has
    #                          a different uncertianty.
    def add(self, X, y, training_sigma=0.0005):
        if not isinstance(training_sigma, Sequence):
            training_sigma = np.ones(len(y)) * training_sigma

        y = y - self.mean_func(X)

        if self.X_train is None:
            self.X_train = X
            self.y_train = y
            self.training_sigma = training_sigma
        else:
            self.X_train = np.append(self.X_train, X, axis=0)
            self.y_train = np.append(self.y_train, y, axis=0)
            self.training_sigma = np.append(self.training_sigma, training_sigma, axis=0)


    ## clear_training
    # clears all training data from the GP
    def reset(self):
        self.X_train = None
        self.y_train = None


    ## Predicts the output of the GP at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X):
        if self.X_train is None:
            self.cov = self.cov_func.cov(X,X)
            sigma = np.diagonal(self.cov)
            # just in case do to numerical instability a negative variance shows up
            sigma = np.maximum(0, sigma)
            return np.zeros(len(X)), sigma

        #### This function treats Y as the training data
        Y = self.X_train
        covXX = self.cov_func.cov(X,X) # covMatrix(X, X, self.cov_func)
        covXY = self.cov_func.cov(X,Y) #covMatrix(X, Y, self.cov_func)
        covYX = np.transpose(covXY)

        error = np.zeros((len(Y), len(Y)))
        np.fill_diagonal(error, self.training_sigma)

        covYY = self.cov_func.cov(Y, Y) + error #covMatrix(Y, Y, self.cov_func) + error


        L = np.linalg.cholesky(covYY)
        v = cho_solve((L, True), covYX)

        muX_Y = np.matmul(covXY, cho_solve((L, True), self.y_train))
        # stored as an instance variable in case it is needed for some reason
        #self.cov = covXX -  np.matmul(np.matmul(covXY, covYYinv), covYX)
        self.cov = covXX - (covXY @ v)

        sigmaX_Y = np.diagonal(self.cov)
        # just in case do to numerical instability a negative variance shows up
        sigmaX_Y = np.maximum(0, sigmaX_Y)

        return muX_Y + self.mean_func(X), sigmaX_Y

