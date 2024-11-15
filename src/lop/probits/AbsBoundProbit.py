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

# AbsBoundProbit.py
# Written Ian Rankin - October 2021
# Modified code by Nicholas Lawerence from here
# https://github.com/osurdml/GPtest/tree/feature/wine_statruns/gp_tools
# Used with permission.
#
# A set of Gaussian Process Probits from
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen


import numpy as np
import scipy.stats as st

from lop.probits import ProbitBase
from lop.probits import std_norm_pdf, std_norm_cdf
from lop.utilities import d_log_pdf_gamma, log_pdf_gamma

from scipy.special import digamma, polygamma
from scipy.stats import norm, beta

import math

from numba import jit

import pdb

# #@jit(nopython=True)
# def numba_beta_pdf1(x, aa, bb):
#     eps = 1e-11
#     B_inv = np.empty(aa.shape, dtype=float)

#     for i in range(aa.shape[0]):
#         B_inv[i] = math.lgamma(aa[i] + bb[i]) - math.lgamma(aa) - math.lgamma(bb)
#         #B_inv[i] = math.gamma(aa[i] + bb[i]) / (math.gamma(aa[i]) * math.gamma(bb[i]))

#     top = (aa-1) * np.log(x) + (bb-1) * np.log(1-x)
#     log_beta = top + B_inv
#     beta = np.exp(log_beta)
#     return beta

# @jit(nopython=True)
# def numba_beta_pdf2(x, aa, bb):
#     B_inv = np.empty(aa.shape, dtype=float)

#     for i in range(aa.shape[0]):
#         for j in range(aa.shape[1]):
#             B_inv[i,j] = math.gamma(aa[i,j] + bb[i,j]) / (math.gamma(aa[i,j]) * math.gamma(bb[i,j]))

#     top = np.power(x, aa-1) * np.power(1-x, bb-1)
#     return top * B_inv

# @jit(nopython=True)
# def numba_beta_pdf3(x, aa, bb):
#     B_inv = np.empty(aa.shape, dtype=float)

#     for i in range(aa.shape[0]):
#         for j in range(aa.shape[1]):
#             for k in range(aa.shape[2]):
#                 B_inv[i,j,k] = math.gamma(aa[i,j,k] + bb[i,j,k]) / (math.gamma(aa[i,j,k]) * math.gamma(bb[i,j,k]))

#     top = np.power(x, aa-1) * np.power(1-x, bb-1)

#     return top * B_inv

# @jit(nopython=True)
# def numba_beta_pdf4(x, aa, bb):
#     B_inv = np.empty(aa.shape, dtype=float)

#     for i in range(aa.shape[0]):
#         for j in range(aa.shape[1]):
#             for k in range(aa.shape[2]):
#                 for l in range(aa.shape[3]):
#                     B_inv[i,j,k,l] = math.gamma(aa[i,j,k,l] + bb[i,j,k,l]) / (math.gamma(aa[i,j,k,l]) * math.gamma(bb[i,j,k,l]))

#     top = np.power(x, aa-1) * np.power(1-x, bb-1)

#     return top * B_inv

# @jit(nopython=True)
# def numba_beta_pdf(x, aa, bb):
#     B_inv = math.gamma(aa + bb) / (math.gamma(aa) * math.gamma(bb))


#     top = np.power(x, aa-1) * np.power(1-x, bb-1)

#     return top * B_inv

## AbsBoundProbit
# This is almost directly Nick's code, for absolute bounded inputs.
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen
#
#
class AbsBoundProbit(ProbitBase):
    type = 'bounded continuous'
    y_type = 'bounded'


    ## Constructor
    # @param sigma - the slope of the probit, basically scales how far away from
    #               0 the latent has to be to to move away from 0.5 output. Sigma should
    #               basically relate to the range of the latent function
    # @param v - the precision, kind of related to inverse of noise, high v is sharp distributions
    # @param eps - epsilon to avoid division by 0 errors.
    def __init__(self, sigma=1.0, v=80.0, optimize_parameters=True, optimize_v_only=True, eps=1e-10):
        self.set_sigma(sigma)
        self.set_v(v)
        
        self.log2pi = np.log(2.0*np.pi)
        self.eps = eps

        self.sigma_k = 1
        self.sigma_theta = 1

        self.v_k = 2.0
        self.v_theta = 25.0

        self.optimize_parameters = optimize_parameters
        self.optimize_v_only = optimize_v_only

    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - a list of hyperparameters [sigma, v]
    #               sigma, the slope of the probit
    #               v, precision, related to inverse of noise
    def set_hyper(self, hyper):
        if self.optimize_parameters:
            if self.optimize_v_only:
                self.set_v(hyper[0])
            else:
                self.set_v(hyper[1])
                self.set_sigma(hyper[0])

    ## get_hyper
    # Gets a numpy array of hyperparameters for the probit
    def get_hyper(self):
        if self.optimize_parameters:
            if self.optimize_v_only:
                return np.array([self.v])
            else:
                return np.array([self.sigma, self.v])
        else:
            return super().get_hyper()

    ## Performs random sampling using the same liklihood function used by the param
    # liklihood function
    # @return numpy array of independent samples.
    def randomize_hyper(self):
        if self.optimize_v_only:
            return np.array([np.random.gamma(self.v_k, self.v_theta)])
        else:
            return np.array([
                np.random.gamma(self.sigma_k, self.sigma_theta),
                np.random.gamma(self.v_k, self.v_theta)])

    ## param_likli
    # log liklihood of the parameter (prior)
    # for this is a parameterized gamma_distribution. Scaled for functions of 
    # approximently size 1 and distance between points in [0,10] ish range
    def param_likli(self):
        if self.optimize_parameters:
            L_v = log_pdf_gamma(self.v, self.v_k, self.v_theta)
            if self.optimize_v_only:
                return L_v
            return log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta) + L_v
                    
        else:
            return super().param_likli()

    ## grad_param_likli
    # gradient of the log liklihood of the parameter (prior)
    # @return numpy array of gradient of each parameter
    def grad_param_likli(self):
        if self.optimize_parameters:
            if self.optimize_v_only:
                return np.array([d_log_pdf_gamma(self.v, self.v_k, self.v_theta)])
            else:
                return np.array([d_log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta),
                        d_log_pdf_gamma(self.v, self.v_k, self.v_theta)])
        else:
            return super().grad_param_likli()

    ## set_sigma
    # Sets the sigma on the absolute bounded probit.
    # Also calculates inverse to the sigma for fast calculation.
    # @param sigma - the slope of the probit, basically scales how far away from
    #               0 the latent has to be to to move away from 0.5 output. Sigma should
    #               basically relate to the range of the latent function
    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2

    ## sets the v variable.
    # @param v - the precision, kind of related to inverse of noise, high v is sharp distributions
    def set_v(self, v):
        self.v = v

    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        print("Beta distribution, probit mean link.")
        print("Sigma: {0:0.2f}, v: {1:0.2f}".format(self.sigma, self.v))


    ## mean_link
    # The mean link function for the probit function.
    # Defined as equation 11 in Section 3.2.1
    # @param F - the predicted locations
    def mean_link(self, F):
        ml = np.empty(F.shape)
        np.clip(std_norm_cdf(F*self._isqrt2sig), self.eps, 1.0-self.eps, out=ml)
        return ml

    ## get_alpha_beta
    # the alpha and beta function for the mean function.
    # Equation 10
    # @param F - the input data
    def get_alpha_beta(self, F):
        ml = self.mean_link(F)
        #aa = self.v * ml
        #bb = self.v - aa    # = self.v * (1-ml)
        return self.get_alpha_beta_ml(F, ml)

    def get_alpha_beta_ml(self, F, ml):
        aa = self.v * ml
        bb = self.v - aa    # = self.v * (1-ml)
        return aa, bb


    ## derivatives
    # Calculates the derivatives of the probit with the given input data
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of P(y|x,theta) with respect to F
    #       py - P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        #y_sel = y[0][y[1]]
        y_sel = y[0]
        f = F[y[1]]

        aa, bb = self.get_alpha_beta(f)

        # Trouble with derivatives...
        dpy_df = self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (np.log(y_sel) - np.log(1-y_sel) - digamma(aa) + digamma(bb))

        Wdiag = - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
            f * self._i2var * (np.log(y_sel) - np.log(1.0-y_sel) - digamma(aa) + digamma(bb)) +
            self.v * self._isqrt2sig * std_norm_pdf(f*self._isqrt2sig) * (polygamma(1, aa) + polygamma(1, bb)) )


        #py = np.log(beta.pdf(y_sel, aa, bb))
        py = beta.logpdf(y_sel, aa, bb)

        # setup the indexing
        full_W = np.zeros((F.shape[0], F.shape[0]))
        full_W[y[1],y[1]] = Wdiag

        full_dpy_df = np.zeros(F.shape[0])
        full_dpy_df[y[1]] = dpy_df

        return -full_W, full_dpy_df, np.sum(py)


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit (np(float) data, np(int)index)
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        return np.exp(self.log_likelihood(y, F))

    def log_likelihood(self, y, F):
        y_selected = y[0]
        f = F[y[1]]
        aa, bb = self.get_alpha_beta(f)

        py = beta.logpdf(y_selected, aa, bb)
        full_py = np.zeros(F.shape[0])
        full_py[y[1]] = py

        return np.sum(full_py)



    ## calc_W_dF
    # Calculate the third derivative of the W matrix.
    # d ln(p(y|F)) / d f_i, f_j, f_k
    # This returns a 3d matrix of (N x N x N) where N is the length of the F vector.
    # Equation (65)
    #
    # @param y - the label for the given probit
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @reutrn 3d matrix
    def calc_W_dF(self, y, F):
        y_sel = y[0]
        f = F[y[1]]

        aa, bb = self.get_alpha_beta(f)

        aa0, bb0 = digamma(aa), digamma(bb)
        aa1, bb1 = polygamma(1, aa), polygamma(1, bb)
        aa2, bb2 = polygamma(2, aa), polygamma(2, bb)

        v = self.v
        sigma = self.sigma
        sqrt2 = np.sqrt(2)

        gauss_pdf = std_norm_pdf(f / (sqrt2*sigma))


        term1_a = -(v / (2*sqrt2*sigma*sigma*sigma)) * gauss_pdf * (1 - ((f*f)/(2*sigma*sigma)))
        term1_b = np.log(y_sel) - np.log(1-y_sel) - aa0 + bb0

        term2_a = ((3*v*v*f) / (4*sigma*sigma*sigma*sigma)) * gauss_pdf * gauss_pdf
        term2_b = aa1 + bb1

        term3_a = (v*v*v) / (2*sqrt2 * sigma*sigma*sigma)
        term3_b = gauss_pdf * gauss_pdf * gauss_pdf * (bb2 - aa2)


        Wdiag = term1_a*term1_b + term2_a*term2_b + term3_a*term3_b

        dW = np.zeros((F.shape[0], F.shape[0], F.shape[0]))
        dW[y[1],y[1],y[1]] = Wdiag

        return dW

    ## calc_W_dHyper
    # Calculate the derivative of the W matrix with respect to hyper parameters.
    # dW / dHyper
    # This returns a 3d matrix of (N x N x N) where N is the length of the F vector.
    # Equation (70)
    #
    # @param y - the label for the given probit
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @reutrn 2d matrix
    def calc_W_dHyper(self, y, F):
        if not self.optimize_parameters:
            return super().calc_W_dHyper(y, F)

        y_sel = y[0]
        f = F[y[1]]

        sigma = self.sigma
        v = self.v
        sqrt2 = np.sqrt(2)
        mu = self.mean_link(f)
        aa, bb = self.get_alpha_beta(f)

        aa0, bb0 = digamma(aa), digamma(bb)
        aa1, aa2 = polygamma(1, aa), polygamma(2, aa)
        bb1, bb2 = polygamma(1, bb), polygamma(2, bb)

        gauss_pdf = std_norm_pdf(f / (np.sqrt(2)*sigma))
        gauss_pdf2 = gauss_pdf * gauss_pdf
        gauss_pdf3 = gauss_pdf * gauss_pdf2

        sigma2 = sigma * sigma
        sigma3 = sigma2 * sigma
        sigma4 = sigma3 * sigma

        tmp = np.log(y_sel) - np.log(1 - y_sel) - aa0 + bb0

        # calculate dW/dv
        # see page 10 of my hand derivations
        term1 = -(f / (2*sqrt2*sigma3)) * gauss_pdf * tmp

        term2_a = -(v*f / (2*sqrt2 * sigma3)) * gauss_pdf
        term2_b = -mu*aa1 + bb1 - mu*bb1

        term3 = (v / sigma2) * gauss_pdf2 * (aa1 + bb1)
        
        term4_a = ((v*v) / (2*sigma2)) * gauss_pdf2
        term4_b = mu*aa1 + (1-mu)*bb2

        dW_dv = term1 + term2_a*term2_b + term3 + term4_a*term4_b

        # dW/dSigma
        # See page 9 of my hand calculations
        term1_a = (v*f / (4 *sqrt2 *sigma4)) * gauss_pdf 
        term1_b = (6 - ((f*f) / (sigma2)))
        term1_c = tmp

        term2_a = ((v*v) / (sigma3)) * gauss_pdf2
        term2_b = 1 + ((-3*f*f) / (4*sigma2))
        term2_c = aa1 + bb1

        term3_a = ((v*v*v*f) / (2*sqrt2*sigma4)) * gauss_pdf3
        term3_b = aa2 - bb2

        dW_dSigma = term1_a*term1_b*term1_c + term2_a*term2_b*term2_c + term3_a*term3_b

        # if (dW_dSigma > 1000).any():
        #     pdb.set_trace()

        if self.optimize_v_only:
            dW_dHyper = np.zeros((1, len(F), len(F)))
            dW_dHyper[0, y[1], y[1]] = dW_dv
        else:
            dW_dHyper = np.zeros((2, len(F), len(F)))
            dW_dHyper[1, y[1], y[1]] = dW_dv
            dW_dHyper[0, y[1], y[1]] = dW_dSigma

        return -dW_dHyper


    ## grad_hyper
    # Calculates the gradient of p(y|F) given the parameters of the probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return numpy array (gradient of probit with respect to hyper parameters)
    def grad_hyper(self, y, F):
        if not self.optimize_parameters:
            return super().grad_hyper(y, F)

        y_sel = y[0]
        f = F[y[1]]

        sigma = self.sigma
        v = self.v

        mu = self.mean_link(f)
        aa, bb = self.get_alpha_beta(f)
        aa0, bb0 = digamma(aa), digamma(bb)

        gauss_pdf = std_norm_pdf(f / (np.sqrt(2)*sigma))

        # dpy_dv
        dig_aa_bb = digamma(aa + bb)
        term1 = mu*np.log(y_sel) + (1 - mu)*np.log(1-y_sel)
        term2 = -mu*aa0 - (1 - mu)*bb0 + dig_aa_bb

        dpy_dv = np.sum(term1 + term2)

        # dpy_dSigma
        mult_term = v*f*gauss_pdf / (np.sqrt(2)*sigma*sigma)
        term = np.log(1 - y_sel) - np.log(y_sel) + aa0 - bb0

        dpy_dSigma = np.sum(mult_term*term)

        #pdb.set_trace()

        if self.optimize_v_only:
            return np.array([dpy_dv])
        return np.array([dpy_dSigma, dpy_dv])


    ## cdf for the beta function.
    def cdf(self, y, F):
        aa, bb = self.get_alpha_beta(F)
        return beta.cdf(y, aa, bb)

