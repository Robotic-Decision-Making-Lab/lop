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


# Model.py
# Written Ian Rankin - December 2023
#
# A base class for all models in lop
# Designed to handle active learning for each model type

import numpy as np

class Model:

    ## constructor
    # constructor for the base model
    def __init__(self, active_learner=None):
        self.active_learner = active_learner
        if self.active_learner is not None:
            self.active_learner.set_model(self)


    def reset(self):
        raise(NotImplementedError("Model reset is not implemented"))
    
    ## Predicts the output of the model at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n), other output data (variance, covariance,etc)
    def predict(self, X):
        raise(NotImplementedError("Model predict is not implemented"))

    ## Predicts the output of the GP at new locations for large
    # numbers of data points.
    # Useful for GP where entire Covariance might not be needed, just mean and variance
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n), other output data (variance, covariance,etc)
    def predict_large(self, X):
        return self.predict(X)

    
    ## () operator
    # This is just a wrapper around the predict function, without
    # any additional variance of other model information
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def __call__(self, X):
        y, other = self.predict(X)
        return y


    ## select_best
    # Returns the best index from the list of candidate_pts
    # @param candidate_pts - list of candidate points to select from.
    #
    # @return index of the best candidate pt
    def select_best(self, candidate_pts):
        y = self.__call__(candidate_pts)
        return np.argmax(y)


    ## select
    # This function calls the active learner and specifies the
    def select(self, candidate_pts, num_alts, prefer_num=-1):
        return self.active_learner.select(candidate_pts, num_alts, prefer_num)


## SimplelestModel
# A simple model that just outputs exactly it's input no matter what
class SimplelestModel(Model):
    ## Predicts the output of the model at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n), other output data (variance, covariance,etc)
    def predict(self, X):
        return X, None

    def clear(self):
        pass



