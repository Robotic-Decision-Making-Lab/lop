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

# PreferenceGP.py
# Written Ian Rankin - December 2023
# Previous code written September 2021
#
# A Gaussian Process implementation that handles ordered pairs of preferences
# for the training data rather than direct absolute samples.
# Essentially optimizes the solution of the samples given to the GP.
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen

import numpy as np
import scipy.optimize as opt
from scipy.linalg import cho_solve
import sys
if sys.version_info[0] >= 3 and sys.version_info[1] >= 3:
    from collections.abc import Sequence
else:
    from collections import Sequence

from lop.models import PreferenceModel
from lop.utilities import k_fold_x_y, get_y_with_idx

import math

import pdb

## PreferenceGP
# A Gaussian Process implementation that handles ordered pairs of preferences
# for the training data rather than direct absolute samples.
# Essentially optimizes the solution of the samples given to the GP.
#
# Based off of the math given in:
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen
class PreferenceGP(PreferenceModel):

    ## constructor
    # @param cov_func - the covariance function to use
    # @param normalize_gp - [opt] normalizes the GP F to -1,1
    # @param pareto_pairs - [opt] specifies whether to consider pareto optimality
    # @param normalize_positive - [opt] normalizes the gp F between 0,1
    # @param other_probits - [opt] allows specification of additional probits
    # @param mat_int - [opt] allows specification of different matrix inversion functions
    #                   defaults to the numpy.linalg.pinv invert function
    # @param use_hyper_optimization - [opt] sets whether optimizatiion should attempt to
    #                   do hyperparameter optimization
    # @param K_sigma - [opt default=0.01] sets the sigma value on the covariance matrix.
    #                   K = cov(X) + I * K_sigma
    # @param active_learner - defines if there is an active learner for this model
    def __init__(self, cov_func, normalize_gp=False, pareto_pairs=False, \
                normalize_positive=False, other_probits={}, mat_inv=np.linalg.pinv, \
                use_hyper_optimization=False, K_sigma = 0.00, active_learner=None):
        super(PreferenceGP, self).__init__(pareto_pairs, other_probits, active_learner)

        self.cov_func = cov_func
        self.invert_function = mat_inv

        self.lambda_gp = 0.9

        self.normalize_gp = normalize_gp
        
        self.normalize_positive = normalize_positive
        self.use_hyper_optimization = use_hyper_optimization

        self.K_sigma = K_sigma
        self.delta_f = 0.0002 # set the convergence to stop
        self.maxloops = 100
        



    ## Predicts the output of the GP at new locations
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n)
    def predict(self, X, X_train=None,F=None, W=None):
        if self.X_train is None:
            cov = self.cov_func.cov(X,X)
            sigma = np.diagonal(cov)
            # just in case do to numerical instability a negative variance shows up
            sigma = np.maximum(0, sigma)
            self.cov = cov
            return np.zeros(len(X)), sigma

        # lazy optimization of GP
        if not self.optimized:
            self.optimize(optimize_hyperparameter=self.use_hyper_optimization)

        X_test = X
        if X_train is None:
            X_train = self.X_train
        if F is None:
            F = self.F
        if W is None:
            W = self.W
        K = self.K
        

        covXX_test = self.cov_func.cov(X_test, X_train)
        covTestTest = self.cov_func.cov(X_test, X_test)

        covX_testX = np.transpose(covXX_test)

        try:
            L = np.linalg.cholesky(K)
        except:
            pdb.set_trace()
        lower = True
        alpha = cho_solve((L, lower), F)

        ####### calculate the mu of the value
        mu = np.matmul(covXX_test, alpha)


        ######### calculate the covariance and the sigma on the covariance
        tmp = self.invert_function(np.identity(len(K)) + np.matmul(W, K))
        tmp2 = np.matmul(covXX_test, tmp)
        tmp3 = np.matmul(W, covX_testX)

        self.cov = covTestTest - np.matmul(tmp2, tmp3)
        sigma = np.diagonal(self.cov)
        sigma = np.maximum(0, sigma)

        return mu, sigma

    ## Predicts the output of the GP at new locations for large
    # numbers of data points.
    # Useful for GP where entire Covariance might not be needed, just mean and variance
    # @param X - the input test samples (n,k).
    #
    # @return an array of output values (n), other output data (variance, covariance,etc)
    def predict_large(self,X):
        # lazy optimization of GP
        if not self.optimized and self.X_train is not None:
            self.optimize(optimize_hyperparameter=self.use_hyper_optimization)

        num_at_a_time = 15

        num_runs = int(math.ceil(X.shape[0] / num_at_a_time))

        mu = np.empty(X.shape[0])
        sigma = np.empty(X.shape[0])

        for i in range(num_runs):
            low_i = i*num_at_a_time
            high_i = min(X.shape[0], low_i+num_at_a_time)

            mu_loc, sigma_loc = self.predict(X[low_i:high_i])
            sigma[low_i:high_i] = sigma_loc
            mu[low_i:high_i] = mu_loc

        return mu, sigma

    ####################### Functions needed for finding the mode of sample outputs


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
    def derivatives(self, y, F):
        W = np.zeros((len(F), len(F)))
        grad_ll = np.zeros(len(F))
        log_likelihood = 0

        for j, probit in enumerate(self.probits):
            if y[j] is not None:
                W_local, dpy_df_local, py_local = probit.derivatives(y[j], F)
                try:
                    W += W_local
                except:
                    pdb.set_trace()
                grad_ll += dpy_df_local
                log_likelihood += py_local

        return W, grad_ll, log_likelihood


    

    


    ## find_mode
    # This function calculates the mode of the F vector by using the damped newton update
    #
    def find_mode(self, x_train, y_train, debug=False):
        X_train = x_train

        self.K = self.cov_func.cov(X_train, X_train) + np.eye(X_train.shape[0]) * self.K_sigma

        F = np.random.random(len(X_train))
        
        # damped newton method
        n_loops = 0
        f_err = self.delta_f + 1

        # checking for convergence by optimization amount
        while f_err > self.delta_f:
            self.W, self.grad_ll, self.log_likelihood = \
                                            self.derivatives(y_train, F)

            #K_inv = self.invert_function(self.K)
            L = np.linalg.cholesky(self.K)
            L_inv = self.invert_function(L)
            K_inv = L_inv.T @ L_inv
            #gradient = self.grad_ll - (K_inv @ F)
            gradient = self.grad_ll - cho_solve((L,True), F)

            # Hessian:
            hess = -self.W - K_inv

            #pdb.set_trace()
            #l_f = lambda F, x, y: -self.likli_f
            F_new = self.newton_update( F, # estimated training values
                                        gradient, # Gradient input to newton's method
                                        hess, # The hessian matrix input
                                        self.likli_f,
                                        (x_train, y_train),
                                        self.invert_function,
                                        lambda_type="binary", # sets the type of line search or static to perform
                                        line_search_max_itr=5)

            # normalize F
            if self.normalize_gp:
                if self.normalize_positive:
                    min = np.amin(F_new)
                    F_new = (F_new - min)
                    max = np.amax(F_new)
                    F_new = F_new / max
                else:
                    F_norm = np.linalg.norm(F_new, ord=np.inf)
                    F_new = F_new / F_norm

            # check for convergence
            f_err = np.linalg.norm((F_new - F), ord=np.inf)
            if debug:
                print("\tf_err="+str(f_err))
            F = F_new


            n_loops += 1
            if n_loops > self.maxloops:
                print('WARNING: maximum loops in find_mode exceeded. Returning current solution')
                break

        if debug:
            print('Optimization ran for: '+str(n_loops))
        
        self.n_loops = n_loops

        self.F = F
        # calculate W with final F
        self.W, self.grad_ll, self.log_likelihood = \
                                        self.derivatives(y_train, self.F)

    def optimize(self, optimize_hyperparameter=False):
        if optimize_hyperparameter:
            k_fold = 2
            splits = k_fold_x_y(self.X_train, self.y_train, k_fold)

            for i in range(k_fold):
                k_fold_but_valid = list(range(k_fold))
                k_fold_but_valid.remove(i)

                train_idxs = []
                for k in k_fold_but_valid:
                    train_idxs += splits[k]

                X_training = self.X_train[train_idxs]
                y_training = get_y_with_idx(self.y_train, train_idxs)
                

                print('Hyperparameters: ')
                print(self.get_hyper())
                self.find_mode(X_training, y_training)
                self.hyperparameter_search(X_training,
                                            y_training,
                                            self.X_train, 
                                            self.y_train, 
                                            i)


        print('Output hyperparameters')
        print(self.get_hyper())
        self.find_mode(self.X_train, self.y_train)
        self.optimized = True

    ## hyperparameter_obj
    # the objective function to be called by scipy optimize
    # @param x - the input hyperparameters (numpy array)
    # @param args - the input arguments either than the actual hyperparameters
    #
    # @return the objective to be minimized
    def hyperparameter_obj(self, x, X_train, y_train, X_valid, y_valid):
        #X_train = args[0]
        #y_valid = args[1]
        self.set_hyper(x)
        #F = self.F
        W, grad_ll, log_py_f = self.derivatives(y_train, self.F)
        F,_ = self.predict(X_valid, X_train, self.F, W)
        
        return -self.likli_f(F, X_valid, y_valid)
        #return self.likli_f_hyper(F, X_valid, y_valid)


    ## hyperparameter_search
    # This function performs an iteration of searching hyperparemters
    def hyperparameter_search(self, X_train, y_train, X_valid, y_valid, itr=0):
        # uses the scipy optimizer to perform the optimization
        self.optimized = True

        self.visualize_hyperparameter(self.F, X_valid, X_train, y_valid, y_train, itr)

        x0 = self.get_hyper()

        args = (X_train, y_train, X_valid, y_valid)
        bounds = [(0.001, 10.0) for i in range(len(x0))]
        result = opt.minimize(
                    fun=self.hyperparameter_obj,
                    x0=x0,
                    bounds=bounds,
                    args=args,
                    tol=0.01, 
                    options={'maxiter': 20, 'disp': False})
                    #jac=)

        print(result)

        self.set_hyper(result.x)




    ## get_hyper
    # get the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @return a numpy array of all hyperparameters (N,)
    def get_hyper(self):
        # get probit hyperparameters
        probit_p = super().get_hyper()

        kernel_p = self.cov_func.get_param()

        return np.append(probit_p, kernel_p, axis=0)

    ## set_hyper
    # set the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @param x - the input hyper parameters as a (n, ) numpy array
    def set_hyper(self, x):
        num_probit_p = len(super().get_hyper())


        #super().set_hyper(x[:num_probit_p])
        super().set_hyper(np.array([0.5]))
        self.cov_func.set_param(x[num_probit_p:])



    ######################## helper functions for calculating liklihood

    ## likli_f_hyper
    # calculates the posterior log liklihood function for the model given parameters
    # This is equation (25)
    # @param F - the given locations of the model given the x and labels y
    # @param x - the given inputs of the function
    # @param y - the labels of given function (pairwise parameters etc)
    #
    # @return a scalar value as the log liklihood of the model
    def likli_f_hyper(self, F, x, y):
        K = self.cov_func.cov(x, x) + np.eye(x.shape[0]) * self.K_sigma

         # calculate the log-likelyhood of the data given F
        #log_py_f = self.log_likelyhood_training(F, y)
        x_prev = super().get_hyper()
        super().set_hyper(np.array([0.1]))
        W, grad_ll, log_py_f = self.derivatives(y, F)
        super().set_hyper(x_prev)
        L = np.linalg.cholesky(K)

        
        term1 = 0.5*(np.transpose(F) @ cho_solve((L, True), F))

        term2 = 0.5 * np.log(np.linalg.det(np.eye(K.shape[0]) + (K @ W)))

        return log_py_f - term1 - term2

    ## likli_f
    # calculates the posterior log liklihood function for the model
    # @param F - the given locations of the model given the x and labels y
    # @param x - the given inputs of the function
    # @param y - the labels of given function (pairwise parameters etc)
    #
    # @return a scalar value as the log liklihood of the model
    def likli_f(self, F, x, y):
        K = self.cov_func.cov(x, x) + np.eye(x.shape[0]) * self.K_sigma

         # calculate the log-likelyhood of the data given F
        log_py_f = self.log_likelyhood_training(F, y)
        L = np.linalg.cholesky(K)

        #K_inv = self.invert_function(K)
        #term1 = 0.5*(np.transpose(F) @ K_inv @ F)
        term1 = 0.5*(np.transpose(F) @ cho_solve((L, True), F))

        # Determinant of lower tringular matrix is product of diagonals
        log_det_K = np.sum(np.log(np.diagonal(L)))
        # det_K = np.linalg.det(K)
        # while det_K <= 0:
        #     #K = K + np.eye(K.shape[0])*0.01
        #     rand_arr = np.random.normal(0, 0.1, size=K.shape[0])
        #     K = K + np.diag(np.where(rand_arr<0, 0, rand_arr))
        #     print('Adding noise to covariance matrix to avoid being singular')
        #     det_K = np.linalg.det(K)
        term2 = log_det_K #np.log(det_K)

        term3 = 0.5*len(F) * np.log(2 * np.pi)

        #log_py_f = 0

        log_prior = log_py_f - term1 - term2 - term3
        return log_prior

   

    ## visualize_hyperparameter
    # Visualize the hyperparameter space for debug purposes
    def visualize_hyperparameter(self, F, x, x_train, y, y_train, itr=0):
        import matplotlib.pyplot as plt

        rbf_sigmas = np.arange(0.001, 15.0, 0.2)
        rbf_lengths = np.arange(0.01,15.0, 0.2)

        liklihoods = np.zeros((rbf_sigmas.shape[0], rbf_lengths.shape[0]))

        X = np.arange(-0.5, 8, 0.1)
        W, _, _ = self.derivatives(y_train, F)
        mu, sigma = self.predict(X, x_train, F, W)
        std = np.sqrt(sigma)

        # perform exhastive search of kernel parameters
        for i, rbf_sigma in enumerate(rbf_sigmas):
            for j, rbf_length in enumerate(rbf_lengths):
                self.cov_func.set_param(np.array([rbf_sigma, rbf_length]))

                W, _, _ = self.derivatives(y_train, F)
                F_post, _ = self.predict(x, x_train, F, W)
                likli = self.likli_f_hyper(F_post, x, y)

                liklihoods[i,j] = likli


        probit_sigmas = np.arange(0.001, 15.0, 0.2)

        liklihoods_pro = np.zeros((rbf_sigmas.shape[0], rbf_lengths.shape[0]))

        # perform exhastive search of kernel lengthscale + probit parameter
        for i, pro_sigma in enumerate(probit_sigmas):
            for j, rbf_length in enumerate(rbf_lengths):
                self.cov_func.set_param(np.array([0.5, rbf_length]))
                self.probits[0].set_sigma(pro_sigma)

                W, _, _ = self.derivatives(y_train, F)
                F_post, _ = self.predict(x, x_train, F, W)
                likli = self.likli_f_hyper(F_post, x, y)

                liklihoods_pro[i,j] = likli


        # plot the data
        plt.figure()
        plt.contour(rbf_sigmas, rbf_lengths, liklihoods.T, levels=20)
        plt.xlabel('RBF sigma values')
        plt.ylabel('RBF length scale')
        plt.title('Postierer liklihood function (log liklihood) itr: ' + str(itr))
        plt.colorbar()

        plt.figure()
        plt.contour(probit_sigmas, rbf_lengths, liklihoods_pro.T, levels=20)
        plt.xlabel('Probit sigma values')
        plt.ylabel('RBF length scale')
        plt.title('Postierer liklihood function (log liklihood) itr: ' + str(itr))
        plt.colorbar()

        

        plt.figure()
        ax = plt.gca()
        ax.plot(X, mu)
        sigma_to_plot = 1

        ax.fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
        self.plot_preference(head_width=0.1, ax=ax, X_train=x_train, y_train=y_train, F=F)
        ax.scatter(x_train, F)

        plt.title('Gaussian Process estimate (1 sigma) itr: ' + str(itr))
        plt.xlabel('x')
        plt.ylabel('y')
