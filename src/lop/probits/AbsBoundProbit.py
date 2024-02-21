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

import pdb

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
    def __init__(self, sigma=1.0, v=10.0, eps=1e-12):
        self.set_hyper([sigma, v])
        self.log2pi = np.log(2.0*np.pi)
        self.eps = eps

        self.sigma_k = 3
        self.sigma_theta = 0.5

        self.v_k = 5
        self.v_theta = 2

    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - a list of hyperparameters [sigma, v]
    #               sigma, the slope of the probit
    #               v, precision, related to inverse of noise
    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])
        self.set_v(hyper[1])

    ## get_hyper
    # Gets a numpy array of hyperparameters for the probit
    def get_hyper(self):
        return np.array([self.sigma, self.v])

    ## param_likli
    # log liklihood of the parameter (prior)
    # for this is a parameterized gamma_distribution. Scaled for functions of 
    # approximently size 1 and distance between points in [0,10] ish range
    def param_likli(self):
        return log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta) + \
                log_pdf_gamma(self.v, self.v_k, self.v_theta)

    ## grad_param_likli
    # gradient of the log liklihood of the parameter (prior)
    # @return numpy array of gradient of each parameter
    def grad_param_likli(self):
        return np.array([d_log_pdf_gamma(self.sigma, self.sigma_k, self.sigma_theta),
                d_log_pdf_gamma(self.v, self.v_k, self.v_theta)])

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
        ml = np.clip(std_norm_cdf(F*self._isqrt2sig), self.eps, 1.0-self.eps)
        return ml

    ## get_alpha_beta
    # the alpha and beta function for the mean function.
    # Equation 10
    # @param F - the input data
    def get_alpha_beta(self, F):
        ml = self.mean_link(F)
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

        # aa1 = polygamma(1, aa)

        # dpy_df = self.v*std_norm_pdf(f*self._isqrt2sig) * (np.log(y_sel) - np.log(1-y_sel) - digamma(aa) + digamma(bb))

        # norm_pdf = std_norm_pdf(f*self._isqrt2sig)
        # Wdiag = -self.v*self.v * norm_pdf * \
        #         (norm_pdf * (aa1 + aa1) + (f / (2 * self.v * self.sigma*self.sigma)) * \
        #         (np.log(y_sel) - np.log(1-y_sel) - digamma(aa) + digamma(bb)))
        Wdiag = - self.v*self._isqrt2sig*std_norm_pdf(f*self._isqrt2sig) * (
            f * self._i2var * (np.log(y_sel) - np.log(1.0-y_sel) - digamma(aa) + digamma(bb)) +
            self.v * self._isqrt2sig * std_norm_pdf(f*self._isqrt2sig) * (polygamma(1, aa) + polygamma(1, bb)) )


        py = np.log(beta.pdf(y_sel, aa, bb))

        # setup the indexing
        full_W = np.zeros((F.shape[0], F.shape[0]))
        #idx_grid = np.meshgrid(y[1],y[1])
        #full_W[tuple(idx_grid)] = W
        full_W[y[1],y[1]] = Wdiag

        full_dpy_df = np.zeros(F.shape[0])
        full_py = np.zeros(F.shape[0])
        full_dpy_df[y[1]] = dpy_df
        full_py[y[1]] = py

        return -full_W, full_dpy_df, np.sum(full_py)


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit (np(float) data, np(int)index)
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        y_selected = y[0]
        f = F[y[1]]
        aa, bb = self.get_alpha_beta(f)

        py = np.log(beta.pdf(y_selected, aa, bb))
        full_py = np.zeros(F.shape[0])
        full_py[y[1]] = py
        
        return np.exp(np.sum(full_py))


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

        v = self.v
        sigma = self.sigma

        gauss_pdf = std_norm_pdf(y_sel / (np.sqrt(2)*sigma))

        mult_term = v*v*v * gauss_pdf
        term1_a = (1 / (2*v*v*sigma*sigma))*(((f*f) / (2*sigma*sigma)) - 1)
        term1_b = np.log(y_sel) - np.log(1-y_sel) - digamma(aa) + digamma(bb)

        term2_a = 3 * f * gauss_pdf / (2 * v * sigma*sigma)
        term2_b = polygamma(1, aa) + polygamma(1, bb)

        term3 = gauss_pdf*gauss_pdf * (polygamma(2, bb) - polygamma(2, aa))

        Wdiag = mult_term * (term1_a*term1_b + term2_a*term2_b + term3)

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
        y_sel = y[0]
        f = F[y[1]]

        sigma = self.sigma
        v = self.v
        aa, bb = self.get_alpha_beta(f)

        aa0, bb0 = digamma(aa), digamma(bb)
        aa1, aa2 = polygamma(1, aa), polygamma(2, aa)
        bb1, bb2 = polygamma(1, bb), polygamma(2, bb)

        du_df = std_norm_pdf(f / (np.sqrt(2)*sigma))
        du2_df2 = - f * du_df / (2 * sigma * sigma)

        du_dSigma = -(f / sigma) * du_df

        # calculate dW/dv
        term1_b = np.log(y_sel) - np.log(1- y_sel) - aa0 + aa0 - aa * aa1 + bb * bb1
        term2_b = aa1 + 0.5*aa*aa2 + bb1 + 0.5*bb*bb2

        dW_dv = du2_df2 * term1_b - 2 * v * du_df * du_df * term2_b


        # dW/dSigma
        term1_b = np.log(y_sel) - np.log(1 - y_sel) - aa0 + bb0
        term1_a = (f / (2 * sigma * sigma * sigma)) * (3 - (f*f / (2 * sigma * sigma))) * du_df
        term1 = term1_a * v * term1_b

        term2_b = aa1 + bb1
        term2 = du2_df2 * du_dSigma * v * v * term2_b

        term3_a = (1 / sigma) * (((f*f) / (2 * sigma * sigma)) - 1) * du_df
        term3 = 2 * term3_a * du_df * v * v * term2_b

        term4 = du_df*du_df * du_dSigma * v*v*v * (aa2 - bb2)

        dW_dSigma = term1 - term2 - term3 - term4

        dW_dHyper = np.zeros((2, len(F), len(F)))
        dW_dHyper[1, y[1], y[1]] = dW_dv
        dW_dHyper[0, y[1], y[1]] = dW_dSigma

        return dW_dHyper


    ## grad_hyper
    # Calculates the gradient of p(y|F) given the parameters of the probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return numpy array (gradient of probit with respect to hyper parameters)
    def grad_hyper(self, y, F):
        y_sel = y[0]
        f = F[y[1]]

        sigma = self.sigma
        v = self.v

        mu = self.mean_link(f)
        aa, bb = self.get_alpha_beta(f)
        aa0, bb0 = digamma(aa), digamma(bb)

        da_dv = mu
        db_dv = 1 - mu

        # dpy_dv
        dig_aa_bb = digamma(aa + bb)
        tmp1 = np.log(y_sel) - aa0 + dig_aa_bb
        tmp2 = np.log(1 - y_sel) - bb0 + dig_aa_bb
        term1 = da_dv * tmp1
        term2 = db_dv * tmp2

        dpy_dv = np.sum(term1 + term2)

        # dpy_dSigma
        du_dSigma = -(f / sigma) * std_norm_pdf(f / (np.sqrt(2)*sigma))
        da_dSigma = v * du_dSigma
        db_dSigma = -da_dSigma

        dpy_dSigma = np.sum(da_dSigma * tmp1 + db_dSigma * tmp2)

        #pdb.set_trace()

        return np.array([dpy_dSigma, dpy_dv])


    ## cdf for the beta function.
    def cdf(self, y, F):
        aa, bb = self.get_alpha_beta(F)
        return beta.cdf(y, aa, bb)

