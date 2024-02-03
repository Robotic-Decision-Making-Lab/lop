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

    ## get_hyper
    # get the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @return a numpy array of all hyperparameters (N,)
    def get_hyper(self):
        raise(NotImplementedError("Model get_hyper is not implemented"))

    ## set_hyper
    # set the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @param x - the input hyper parameters as a (n, ) numpy array
    def set_hyper(self, x):
        raise(NotImplementedError("Model set_hyper is not implemented"))

    ## grad_hyper
    # get the gradient of the hyperparameters
    #
    # @return (n,) numpy array of gradient of each hyperparameter for the model
    def grad_hyper(self):
        raise(NotImplementedError("Model grad_hyper is not implemented"))


    ## select
    # This function calls the active learner and specifies the number of alternatives to select
    # A wrapper around calling model.active_learner.select
    # @param candidate_pts - a numpy array of points (nxk), n = number points, k = number of dimmensions
    # @param num_alts - the number of alterantives to selec (including the highest mean)
    # @param prev_selection - [opt, default = []]a list of indicies that 
    # @param prefer_num - [default = None] the points at the start of the candidates
    #                   to prefer selecting from. Returned as:
    #                   a. A number of points at the start of canididate_pts to prefer
    #                   b. A set of points to prefer to select.
    #                   c. 'pareto' to indicate 
    #                   d. Enter 0 explicitly ignore selections
    #                   e. None (default) assumes 0 unless default to pareto is true.
    # @param return_not_selected - [opt default-false] returns the not selected points when there
    #                   a preference to selecting to certian points. [] if not but set to true.
    #                   
    #
    # @return [highest_mean, highest_selection, next highest selection, ...],
    #          selection values for candidate_pts,
    #          only returns highest mean if "always select best is set"
    def select(self, candidate_pts, num_alts, prev_selection=[], prefer_pts=None, return_not_selected=False):
        return self.active_learner.select(candidate_pts, num_alts, prev_selection, prefer_pts, return_not_selected)

## SimplelestModel
# A simple model that just outputs exactly it's input no matter what
class SimplelestModel(Model):
    ## Predicts the output of the model at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n), other output data (variance, covariance,etc)
    def predict(self, X):
        if len(X.shape) > 1:
            return np.sum(X, axis=1), None
        else:
            return X, None

    def reset(self):
        pass



