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

# PreferenceModel.py
# Written Ian Rankin - November 2023
#
# Base set of code for adding preference data.
# 

import numpy as np
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence
from lop.models import Model
from lop.utilities import get_dk
from lop.probits import PreferenceProbit, AbsBoundProbit, OrdinalProbit


class PreferenceModel(Model):

    # init function to setup all needed varaibles
    # @param pareto_pairs - [opt] sets whether adding point should include pareto pairs
    # @param other_probits - [opt] sets additional probit functions for the preference model
    # @param active_learner - [opt] set the active learner for the model.
    def __init__(self, pareto_pairs=False, other_probits={}, active_learner=None):
        super(PreferenceModel, self).__init__(active_learner)
        self.optimized = False

        self.pareto_pairs = pareto_pairs
        self.probits = [PreferenceProbit(sigma = 0.5), OrdinalProbit(), AbsBoundProbit()]
        self.probit_idxs = {'relative_discrete': 0, 'ordinal': 1, 'abs': 2}

        i = 1
        for key in other_probits:
            self.probit_idxs[key] = i
            i += 1

        self.y_train = [None] * len(self.probit_idxs)
        self.X_train = None

        self.prior_idx = None
        self.n_loops = -1


    ## add_prior
    # this function adds prioir data to the GP if desired. Desigend to work with
    # the pareto_pairs constraint to generate a function that ensures pareto_pairs
    # @param bounds - the bounds for the prior pts numpy array (nxn)
    # @param num_pts - the number of prior pts to add
    def add_prior(self, bounds = np.array([[0,1],[0,1]]), num_pts = 100, \
                    method='random', pts=None):
        if self.pareto_pairs == False:
            print('Asked to add prior information without setting pareto pairs to be used. Not adding prior points')
            return

        scaler = bounds[:,1] - bounds[:,0]
        bias = bounds[:,0]

        if method == 'random':
            pts = np.random.random((num_pts, bounds.shape[0])) * scaler + bias

            # replace 2 of the points with the min a max of the prior bounds
            if num_pts > 2:
                pts[0] = bounds[:,0]
                pts[1] = bounds[:,1]

            print(pts)

        elif method == 'exact':
            pts = pts
            num_pts = pts.shape[0]

        if self.X_train is not None:
            self.prior_idx = (self.X_train.shape[0], self.X_train.shape[0]+num_pts)
        else:
            self.prior_idx = (0, num_pts)
        self.add(pts, [], type='relative_discrete')
        self.remove_without_reference()



    ## This function removes all training points with no references
    # This is used because prior points can have no references and cause problems
    # during optimization because of it.
    #
    # @post - X_train has removed indicies, all references in y_train have been
    #          decremented
    def remove_without_reference(self, remove_prior=True):
        counts = np.zeros(self.X_train.shape[0])

        # iterate through each type of training data
        for type in self.probit_idxs:
            y = self.y_train[self.probit_idxs[type]]
            if type == 'relative_discrete':
                for pair in y:
                    counts[pair[1]] += 1
                    counts[pair[2]] += 1

        # check which pts don't have any counts
        #idx_to_rm = [x for x in range(len(counts)) if counts[x] == 0]
        idx_to_rm = []
        cur_cts = 0
        for i in range(len(counts)):
            if counts[i] == 0:
                idx_to_rm.append(i)
                cur_cts += 1

            counts[i] = cur_cts

        # remove X_train points to remove
        self.X_train = np.delete(self.X_train, idx_to_rm, axis=0)

        # reduce indicies of y_train to match removed indicies
        for type in self.probit_idxs:
            y = self.y_train[self.probit_idxs[type]]
            if type == 'relative_discrete':
                for pair in y:
                    pair[1] -= counts[pair[1]]
                    pair[2] -= counts[pair[2]]


        if remove_prior:
            # update the index if they have been removed
            prior_idx = (self.prior_idx[0], self.prior_idx[1] - len(idx_to_rm))
            self.prior_idx = prior_idx



    ## get_prior_pts
    # get the set of prior points if they exist
    # @return numpy array of X_train if it exists, None otherwise
    def get_prior_pts(self):
        if self.prior_idx is not None:
            return self.X_train[self.prior_idx[0]:self.prior_idx[1]]
        else:
            return None

    
    ## reset
    # This function resets all points for the GP
    def reset(self):
        self.y_train = [None] * len(self.probit_idxs)
        self.X_train = None
        self.prior_idx = None

    ## add_training
    # adds training data to the gaussian process
    # appends the data if there already is some training data
    # @param X - the input training data
    # @param y - list of discrete pairs [(dk, uk, vk), ...]
    #                       dk = -1 if u > v, dk = 1 if v > u
    #                       uk = index of the input (for the input set of points)
    #                       vk = index of the second input
    #                       @NOTE That this function updates the indicies if there
    #                       is already training data.
    #                       If inputing ordinal or abs data, it should be a vector of the same
    #                       length as the input data (one y for each x)
    # @param type - type of input ['relative_discrete', 'ordinal', 'abs']
    # @param training_sigma - [opt] sets the uncertianty in the training data
    #                          accepts scalars or a vector if each sample has
    #                          a different uncertianty.
    def add(self, X, y, type='relative_discrete', training_sigma=0):
        if not isinstance(training_sigma, Sequence):
            training_sigma = np.ones(len(y)) * training_sigma

        if self.X_train is None:
            self.X_train = X
            len_X = 0
        else:
            len_X = len(self.X_train)
            self.X_train = np.append(self.X_train, X, axis=0)

        if type == 'relative_discrete':
            if y == []:
                pass
            elif self.y_train[self.probit_idxs[type]] is None:
                self.y_train[self.probit_idxs[type]] = np.array(y)
            else:
                # reset index of pairwise comparisons
                y = [(d, u+len_X, v+len_X) for d, u, v in y]

                self.y_train[self.probit_idxs[type]] = \
                    np.append(self.y_train[self.probit_idxs[type]], np.array(y), axis=0)
        elif type == 'ordinal':
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            if len(y.shape) == 1:
                new_y = np.empty((y.shape[0], 2), dtype=int)
                new_y[:,0] = y
                new_y[:,1] = np.arange(0, y.shape[0])
            else:
                new_y = y

            if (new_y[:,0] <= 0).any():
                raise Exception("Can't pass ordinal with 0 rating (must all be positive ordinal values)")

            if self.y_train[self.probit_idxs[type]] is None:
                self.y_train[self.probit_idxs[type]] = (new_y[:,0], new_y[:,1])
            else:
                # restet index for matching to X_train
                new_y[:,1] += len_X
                old_v = self.y_train[self.probit_idxs[type]][0]
                old_idx = self.y_train[self.probit_idxs[type]][1]

                self.y_train[self.probit_idxs[type]] = \
                    (np.append(old_v, new_y[:,0]), np.append(old_idx, new_y[:,1]))
        elif type == 'abs':
            if isinstance(y, tuple):
                v = y[0]
                idxs = y[1]
            elif isinstance(y, np.ndarray):
                v = y
                idxs = np.arange(len_X, y.shape[0]+len_X)
            else:
                print('abs type received unknown type for y')
                return

            if self.y_train[self.probit_idxs[type]] is not None:
                v = np.append(self.y_train[self.probit_idxs[type]][0], v, axis=0)
                idxs = np.append(self.y_train[self.probit_idxs[type]][1], idxs, axis=0)

            self.y_train[self.probit_idxs[type]] = (v, idxs)


        if self.pareto_pairs:
            pairs = []
            d_better = get_dk(1,0)
            # Go through each new sample and check if it pareto optimal to others
            for i, x in enumerate(X):
                dominate = np.all(x > self.X_train, axis=1)

                cur_pairs = [(d_better, i+len_X, j) for j in range(len(dominate)) if dominate[j]]
                pairs += cur_pairs

            if self.y_train[self.probit_idxs['relative_discrete']] is None:
                self.y_train[self.probit_idxs['relative_discrete']] = np.array(pairs)
            else:
                # only add pairs if there is any pareto pairs to add.
                if len(pairs) > 0:
                    self.y_train[self.probit_idxs['relative_discrete']] = \
                        np.append(self.y_train[self.probit_idxs['relative_discrete']], \
                                    np.array(pairs), axis=0)
                else:
                    import pdb
                    pdb.set_trace()
        # end if for pareto_pairs


        self.optimized = False


    # calculate the log_likelyhood of the training data.
    # log p(Y|F)
    def log_likelyhood_training(self, F):
        log_p_w = 0.0
        for j, probit in enumerate(self.probits):
            if self.y_train[j] is not None:
                p_w_local = probit.log_likelihood(self.y_train[j], F)

                log_p_w += p_w_local

        return log_p_w

    ## calculates the loss function of the log liklihood with prior
    # this is equation (139)
    # @param F - the estimated training values
    def loss_func(self, F):
        raise NotImplementedError("PreferenceModel loss_func is not implemented")

    ## optimize
    # Runs the optimization step required by the user preference GP.
    # @param optimize_hyperparameter - [opt] sets whether to optimize the hyperparameters
    def optimize(self, optimize_hyperparameter=False):
        raise NotImplementedError("PreferenceModel optimize function not implemented")
        self.optimized = True



    ###################### optimization functions for preference models

    ## newton_update
    # This function runs a damped newton update to calculate the next step of the
    # optimization.
    # Can be used with or without line search for selecting the lambda function
    # @param F - the current estimate of the parameters
    # @param K - the current covariance matrix
    # @param W - the Hessian of the loss function
    # @param grad_ll - the gradient of the loss function
    # @param invert_function - [opt] the matrix inversion function to use
    # @param lambda_type - [opt default "static"] sets the type of lambda search, options include ("static", "binary", "iter")
    # @param line_search_max_itr - [opt (5)] line_search_max_itr
    #
    # @return F_new after the update function
    def newton_update(self, F, gradient, hess,
                                invert_function=np.linalg.inv,
                                lambda_type="static",
                                line_search_max_itr = 5):
        # First calculate the descent direction using the gradient and hessian.
        # gradient:
        

        # positive since we are searching for the max.
        descent = gradient @ invert_function(hess)

        if lambda_type == "binary":
            # search along the descent direction for the min point.
            lamb = self.binary_line_search(F, descent, max_itr=line_search_max_itr)
        elif lambda_type == "iter":
            lamb = self.iterative_line_search(F, descent, max_itr=line_search_max_itr)
        elif lambda_type == "static":
            lamb = self.lambda_gp
        else:
            raise ValueError("newton update given bad lambda type of: " + str(lambda_type))

        F_new = F - lamb * descent
        return F_new


    ## binary_line_search
    # performs a binary line search by splitting the search
    # along the descent direction to find the best location.
    # @param F - the input parameters
    # @param descent - the desecent direction scaled to the taylor polynomial
    # @param min_lambda - [opt] the minumum lambda location to search (0.01 default)
    # @param max_lambda - [opt] the maxmimum lambda to search (1.5, default)
    # @param max_itr - [opt] the maximum number of iterations allowed to search
    #
    # @return lambda for the binary search.
    def binary_line_search(self, F, descent, min_lambda=0.0, max_lambda=1.5, max_itr=5):
        # search along the descent direction for the min point.

        #max_lamda = 1.5 # set the max value away from the decent direction to search
        # sets the minimum value to search for
        #min_lamda = 0.001

        for i in range(max_itr):
            lam_dis = max_lambda - min_lambda
            mid_lambda = (lam_dis*0.5 + min_lambda)
            lam_1 = lam_dis * 0.25 + min_lambda
            lam_2 = lam_dis * 0.75 + min_lambda

            loss_1 = self.loss_func(F - lam_1 * descent)
            loss_2 = self.loss_func(F - lam_2 * descent)

            if loss_1 > loss_2:
                max_lambda = mid_lambda
            else:
                min_lambda = mid_lambda
        
        # return the mid point between the search values.
        return 0.5 * (min_lambda + max_lambda)
                
    # @param F - the input parameters
    # @param descent - the desecent direction scaled to the taylor polynomial
    # @param min_lambda - [opt] the minumum lambda location to search (0.01 default)
    # @param max_lambda - [opt] the maxmimum lambda to search (1.5, default)
    # @param max_itr - [opt] the maximum number of iterations allowed to search
    #
    # @return lambda for the binary search.
    def iterative_line_search(self, F, descent, min_lambda=0.01, max_lambda=1.5, max_itr=10):
        lambda_search_pts = np.arange(min_lambda, max_lambda, (max_lambda - min_lambda)/max_itr)
        best_lamb = -1
        best_loss = -np.inf

        for lamb in lambda_search_pts:
            loss = self.loss_func(F - lamb * descent)
            if loss > best_loss:
                best_lamb = lamb
                best_loss = loss

        if best_lamb == -1:
            raise Exception("Iterative line search failed to get a non infinite loss value")

        return best_lamb

