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
from types import SimpleNamespace
import matplotlib.pyplot as plt

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
    # @param active_learner - defines if there is an active learner for this model
    def __init__(self, cov_func, normalize_gp=False, pareto_pairs=False, \
                normalize_positive=False, other_probits={}, mat_inv=np.linalg.pinv, \
                use_hyper_optimization=False, active_learner=None):
        super(PreferenceGP, self).__init__(pareto_pairs, other_probits, active_learner)

        self.cov_func = cov_func
        self.invert_function = mat_inv

        self.lambda_gp = 0.9

        self.normalize_gp = normalize_gp
        
        self.normalize_positive = normalize_positive
        self.use_hyper_optimization = use_hyper_optimization

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

        self.K = self.cov_func.cov(X_train, X_train)

        F = np.random.random(len(X_train))
        
        # damped newton method
        n_loops = 0
        f_err = self.delta_f + 1

        # checking for convergence by optimization amount
        while f_err > self.delta_f:
            self.W, self.grad_ll, self.log_likelihood = \
                                            self.derivatives(y_train, F)

            L = np.linalg.cholesky(self.K)
            L_inv = self.invert_function(L)
            K_inv = L_inv.T @ L_inv
            
            gradient = self.grad_ll - cho_solve((L,True), F)

            # Hessian:
            hess = -self.W - K_inv

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
        if optimize_hyperparameter and self.X_train is not None:
            k_fold = min(math.floor(len(self.X_train) / 2), 2)
            num_iterations = 100

            for j in range(math.ceil(num_iterations / k_fold)):
                splits = k_fold_x_y(self.X_train, self.y_train, k_fold)
                if splits is not None:
                    for i in range(k_fold):
                        k_fold_but_valid = list(range(k_fold))
                        k_fold_but_valid.remove(i)

                        train_idxs = []
                        for k in k_fold_but_valid:
                            train_idxs += splits[k]

                        X_training = self.X_train[train_idxs]
                        y_training = get_y_with_idx(self.y_train, train_idxs)
                        

                        # print('Hyperparameters: ')
                        # print(self.get_hyper())
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
    def hyperparameter_obj(self, x, X_train, y_train, X_valid, y_valid, bounds):

        self.set_hyper(x)
        self.find_mode(X_train, y_train)
        W, grad_ll, log_py_f = self.derivatives(y_train, self.F)
        F,_ = self.predict(X_valid, X_train, self.F, W)
        
        #return x[0]**2 + x[1]**2 + x[2]**2
        #return -self.likli_f(F, X_valid, y_valid)
        #bounds_cost = 5*sum([np.exp(-20*(np.abs(x[i] - bounds[i][0]))) + np.exp(-20*(np.abs(x[i] - bounds[i][1]))) for i in range(len(bounds))])
        return -self.likli_f_hyper(F, X_valid, y_valid) - self.hyper_liklihood()

    def hyperparamter_obj_grad(self, x, X_train, y_train, X_valid, y_valid, bounds):
        self.set_hyper(x)
        self.find_mode(X_train, y_train)
        W, grad_ll, log_py_f = self.derivatives(y_train, self.F)
        F,_ = self.predict(X_valid, X_train, self.F, W)

        return -self.grad_likli_f_hyper(F, X_valid, y_valid) - self.grad_hyper_liklihood()

    ## hyperparameter_search
    # This function performs an iteration of searching hyperparemters
    def hyperparameter_search(self, X_train, y_train, X_valid, y_valid, itr=0):
        # uses the scipy optimizer to perform the optimization
        self.optimized = True

        x0 = self.get_hyper()

        

        #self.set_hyper(x0)
        #x0 = np.random.random((2,))*3.0

        self.debug_print = True

        bounds = [(0.01, 10.0) for i in range(len(x0))]
        #print('cost_prior to update: ' + str(self.hyperparameter_obj(x0, X_train, y_train, X_valid, y_valid, bounds)))

        
        grad_hyper = self.hyperparamter_obj_grad(x0, X_train, y_train, X_valid, y_valid, bounds)
        #grad_hyper[1] = 0
        #print('grad_hyper = ' + str(grad_hyper))

        x_new = x0 - grad_hyper * 0.01
        #print('cost post update: ' + str(self.hyperparameter_obj(x_new, X_train, y_train, X_valid, y_valid, bounds)))
        result = SimpleNamespace(x=x_new)

        # args = (X_train, y_train, X_valid, y_valid, bounds)
        # result = opt.minimize(
        #             fun=self.hyperparameter_obj,
        #             jac=self.hyperparamter_obj_grad,
        #             x0=x0,
        #             bounds=bounds,
        #             args=args,
        #             tol=0.01, 
        #             options={'maxiter': 1, 'disp': False})
        # result = opt.differential_evolution(
        #             func=self.hyperparameter_obj,
        #             bounds=bounds,
        #             args=args,
        #             tol=0.1,
        # )
        self.debug_print = False

        #print(result)

        self.set_hyper(result.x)
        # self.visualize_hyperparameter(self.F, X_valid, X_train, y_valid, y_train, itr)
        # self.set_hyper(result.x)




    ## get_hyper
    # get the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @return a numpy array of all hyperparameters (N,)
    def get_hyper(self):
        # get probit hyperparameters
        probit_p = super().get_hyper()

        kernel_p = self.cov_func.get_param()
        #return np.array([probit_p[0], kernel_p[1]])

        return np.append(probit_p, kernel_p, axis=0)

    ## set_hyper
    # set the hyperparameters for the given model.
    # Particularly intended for hyperparameter optimization.
    #
    # @param x - the input hyper parameters as a (n, ) numpy array
    def set_hyper(self, x):
        num_probit_p = len(super().get_hyper())


        super().set_hyper(x[:num_probit_p])
        #super().set_hyper(np.array([x[0]]))
        self.cov_func.set_param(x[num_probit_p:])

        #self.cov_func.set_param(np.array([0.5, x[1]]))

    def hyper_liklihood(self):
        cov_likli = self.cov_func.param_likli()
        probit_likli = super().hyper_liklihood()

        return cov_likli + probit_likli

    def grad_hyper_liklihood(self):
        d_cov_likli = self.cov_func.grad_param_likli()
        d_probit_likli = super().grad_hyper_liklihood()

        return np.append(d_probit_likli, d_cov_likli, axis=0)

    ######################## helper functions for calculating liklihood


    ## grad_likli_f_hyper
    # This calculates the gradient of the liklihood function and the cost
    # of the function at the same time.
    #
    def grad_likli_f_hyper(self, F, x, y):
        K = self.cov_func.cov(x, x)
        dK_param = self.cov_func.cov_gradient(x,x)

        W, grad_ll, log_py_f = self.derivatives(y, F)

        L_K = np.linalg.cholesky(K)
        alpha_K = cho_solve((L_K, True), F)

        B = np.eye(K.shape[0]) + (K @ W)
        B_inv = self.invert_function(B)

        # Calculate derivative of W matrix with respect to the F vecotr (3d np array)
        dW_f = None
        for i, probit in enumerate(self.probits):
            if y[i] is not None:
                dW_f_local = probit.calc_W_dF(y[i], F)
                if dW_f is None:
                    dW_f = dW_f_local
                else:
                    dW_f += dW_f_local

        

        # Covariance function deriviatives
        dL_cov_f = np.zeros(len(dK_param))
        for i in range(len(dK_param)):
            # equation (204, first half)
            termA_1 = 0.5 * np.transpose(alpha_K) @ dK_param[i] @ alpha_K
            # Equation (204, second half)
            termA_2 = 0.5 * np.trace(B_inv @ dK_param[i] @ W)

            # Equation (205)
            termB_2 = B_inv @ dK_param[i] @ grad_ll
            # Equation 209
            termB_1 = 0.5 * np.trace(B_inv @ K @ dW_f, axis1=1, axis2=2)

            dL_cov_f[i] = termA_1 - termA_2 + np.sum(termB_2 - termB_1)


        # Liklihood function parameters
        probit_grads = []
        for i, probit in enumerate(self.probits):
            if y[i] is not None:
                grad_theta = probit.grad_hyper(y[i], F)
                dW_hyper = probit.calc_W_dHyper(y[i], F)

                # equation (22)
                term2 = 0.5 * np.trace(B_inv @ K @ dW_hyper, axis1=1, axis2=2)

                probit_grads.append(grad_theta-term2)

        grad_hyper = None
        for probit_grad in probit_grads:
            if grad_hyper is None:
                grad_hyper = probit_grad
            else:
                grad_hyper = np.append(grad_hyper, probit_grad, axis=0)
        if grad_hyper is None:
            grad_hyper = dL_cov_f
        else:
            grad_hyper = np.append(grad_hyper, dL_cov_f, axis=0)

        return grad_hyper


    ## likli_f_hyper
    # calculates the posterior log liklihood function for the model given parameters
    # This is equation (25)
    # @param F - the given locations of the model given the x and labels y
    # @param x - the given inputs of the function
    # @param y - the labels of given function (pairwise parameters etc)
    #
    # @return a scalar value as the log liklihood of the model
    def likli_f_hyper(self, F, x, y):
        K = self.cov_func.cov(x, x)

         # calculate the log-likelyhood of the data given F
        #log_py_f = self.log_likelyhood_training(F, y)
        #x_prev = super().get_hyper()
        #super().set_hyper(np.array([0.1]))
        W, grad_ll, log_py_f = self.derivatives(y, F)
        #super().set_hyper(x_prev)
        L = np.linalg.cholesky(K)

        
        term1 = 0.5*(np.transpose(F) @ cho_solve((L, True), F))

        tmp = np.eye(K.shape[0]) + (K @ W)
        term2 = 0.5 * np.log(np.linalg.det(tmp))
        #pdb.set_trace()
        if self.debug_print:
            print(np.linalg.det(tmp))

        return log_py_f - term1 - term2
        #return -term2


    ## likli_f
    # calculates the posterior log liklihood function for the model
    # @param F - the given locations of the model given the x and labels y
    # @param x - the given inputs of the function
    # @param y - the labels of given function (pairwise parameters etc)
    #
    # @return a scalar value as the log liklihood of the model
    def likli_f(self, F, x, y):
        K = self.cov_func.cov(x, x)
         # calculate the log-likelyhood of the data given F
        log_py_f = self.log_likelyhood_training(F, y)
        L = np.linalg.cholesky(K)

        term1 = 0.5*(np.transpose(F) @ cho_solve((L, True), F))

        # Determinant of lower tringular matrix is product of diagonals
        log_det_K = np.sum(np.log(np.diagonal(L)))
        term2 = log_det_K #np.log(det_K)

        term3 = 0.5*len(F) * np.log(2 * np.pi)

        #log_py_f = 0

        log_prior = log_py_f - term1 - term2 - term3
        return log_prior

   

    ## visualize_hyperparameter
    # Visualize the hyperparameter space for debug purposes
    def visualize_hyperparameter(self, F, x, x_train, y, y_train, itr=0):
        import matplotlib.pyplot as plt

        rbf_sigmas = np.logspace(0.01, 1.0, 50)
        rbf_lengths = np.logspace(0.01,1.0, 50)

        liklihoods = np.zeros((rbf_sigmas.shape[0], rbf_lengths.shape[0]))

        X = np.arange(-0.5, 8, 0.1)
        W, _, _ = self.derivatives(y_train, F)
        mu, sigma = self.predict(X, x_train, F, W)
        std = np.sqrt(sigma)

        # perform exhastive search of kernel parameters
        # for i, rbf_sigma in enumerate(rbf_sigmas):
        #     for j, rbf_length in enumerate(rbf_lengths):
        #         self.cov_func.set_param(np.array([rbf_sigma, rbf_length]))

        #         param = self.get_hyper()
        #         param[1] = rbf_sigma
        #         param[2] = rbf_length
        #         likli = self.hyperparameter_obj(param, x_train, y_train, x, y)

        #         liklihoods[i,j] = likli


        probit_sigmas = np.logspace(0.01, 1.0, 50)

        liklihoods_pro = np.zeros((rbf_sigmas.shape[0], rbf_lengths.shape[0]))

        # perform exhastive search of kernel lengthscale + probit parameter
        for i, pro_sigma in enumerate(probit_sigmas):
            for j, rbf_length in enumerate(rbf_lengths):
                self.cov_func.set_param(np.array([0.5, rbf_length]))
                self.probits[0].set_sigma(pro_sigma)

                #param = np.array([pro_sigma, 0.5, rbf_length])
                param = np.array([pro_sigma, rbf_length])
                likli = self.hyperparameter_obj(param, x_train, y_train, x, y)

                liklihoods_pro[i,j] = likli


        # plot the data
        # plt.figure()
        # plt.contour(rbf_sigmas, rbf_lengths, liklihoods.T, levels=20)
        # plt.xlabel('RBF sigma values')
        # plt.ylabel('RBF length scale')
        # plt.title('Postierer liklihood function (log liklihood) itr: ' + str(itr))
        # plt.colorbar()

        plt.figure()
        ax = plt.gca()
        plt.contour(probit_sigmas, rbf_lengths, liklihoods_pro.T, levels=20)
        plt.xlabel('Probit sigma values')
        plt.ylabel('RBF length scale')
        plt.title('Postierer liklihood function (log liklihood) itr: ' + str(itr))
        plt.colorbar()

        ax.set_yscale('log')
        ax.set_xscale('log')
        

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
