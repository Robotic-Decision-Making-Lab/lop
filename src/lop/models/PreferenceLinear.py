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

# PreferenceLinear.py
# Written Ian Rankin - November 2023
#
# A linear latent function to learn the given preferences.

import numpy as np
import pdb

from lop.models import PreferenceModel

class PreferenceLinear(PreferenceModel):
    ## init function
    # @param pareto_pairs - [opt] specifies whether to consider pareto optimality
    # @param other_probits - [opt] allows specification of additional probits
    # @param mat_int - [opt] allows specification of different matrix inversion functions
    #                   defaults to the numpy.linalg.pinv invert function
    # @param active_learner - defines if there is an active learner for this model
    def __init__(self, pareto_pairs=False, other_probits={},
                        mat_inv=np.linalg.pinv, active_learner=None):
        super(PreferenceLinear, self).__init__(pareto_pairs, other_probits,active_learner)

        self.mat_inv = mat_inv
        self.delta_f = 0.002
        self.maxloops = 100


    ## Predicts the output of the linear model at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X):
        # lazy optimization of the model
        if not self.optimized and self.X_train is not None:
            self.optimize()
        elif self.X_train is None:
            if len(X.shape) == 1:
                print('Only 1 reward parameter... linear model practically does not make sense')
                raise Exception("PreferenceLinear can't optimize a single reward value (just scales it)")
            w = np.random.random(X.shape[1])
            self.w = w / np.linalg.norm(w, ord=2)

        F = (X @ self.w[:,np.newaxis])[:,0]
        return F, None

    

    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        if len(self.X_train.shape) > 1:
            w = np.random.random(self.X_train.shape[1])
            w = w / np.linalg.norm(w, ord=2)
        else:
            print('Only 1 reward parameter... linear model practically does not make sense')
            raise Exception("PreferenceLinear can't optimize a single reward value (just scales it)")
            #self.w = np.random.random(1)


        
        # damped newton method        
        w_err = self.delta_f + 1
        n_loops = 0
        while w_err > self.delta_f and n_loops < self.maxloops:
            # damped newton update (to find max, hence plus sign rather than negative sign.)
            W, dpy_dw, py = self.derivatives(self.X_train, self.y_train, w)

            gradient = dpy_dw
            hess = -W

            w_new = self.newton_update(w, # input value to change
                                       gradient=gradient, 
                                       hess=hess,
                                       loss_func = self.loss_func,
                                       lambda_type="binary", # sets the type of line search ("static", "binary", "iter")
                                       line_search_max_itr=5)

            # normalize the weights
            w_new = w_new / np.linalg.norm(w_new, ord=2)

            # measure error convergence
            w_err = np.linalg.norm(w_new - w, ord=2)
            
            w = w_new
            n_loops += 1

        self.n_loops = n_loops
        self.w = w
        self.optimized = True


    ########## Helper functions

    ## derivatives
    # Calculates the derivatives for all of the given probits.
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of log P(y|x,theta) with respect to F
    #       py - log P(y|x,theta) for the given probit
    def derivatives(self, x, y, w):
        F = (x @ w[:,np.newaxis])[:,0]

        W = np.zeros((len(F), len(F)))
        grad_ll = np.zeros(len(F))
        log_likelihood = 0
        for j, probit in enumerate(self.probits):
            if self.y_train[j] is not None:
                W_local, dpy_df_local, py_local = probit.derivatives(y[j], F)

                W += W_local
                grad_ll += dpy_df_local
                log_likelihood += py_local


        # need to multiply by derivative of dl/df * df/dw
        grad_ll = (grad_ll[np.newaxis,:] @ x)[0]
        W = x.T @ W @ x

        return W, grad_ll, log_likelihood


    ## calculates the loss function of the log liklihood with prior
    # this is equation (139)
    # @param w - the weights of the function
    def loss_func(self, w):
        if self.X_train is None:
            return 0
        w = w / np.linalg.norm(w, ord=2)
        F = (self.X_train @ w[:,np.newaxis])[:,0]
        return self.log_likelyhood_training(F, self.y_train)
