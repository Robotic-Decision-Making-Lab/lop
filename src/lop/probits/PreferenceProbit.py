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

# PreferenceProbit.py
# Written Ian Rankin - October 2021
# Modified code by Nicholas Lawerence from here
# https://github.com/osurdml/GPtest/tree/feature/wine_statruns/gp_tools
# Used with permission.
#
# A set of different Gaussian Process Probits from
# Pairwise Judgements and Absolute Ratings with Gaussian Process Priors
#  - a collection of technical details (2014)
# Bjorn Sand Jenson, Jens Brehm, Nielsen


import numpy as np
import scipy.stats as st
from lop.probits import ProbitBase
from lop.probits import std_norm_pdf, std_norm_cdf, calc_pdf_cdf_ratio






## PreferenceProbit
# A relative discrete probit
# Partially taken from Nick's code
# this calculates the probability of preference labels from the latent space
class PreferenceProbit(ProbitBase):
    type = 'preference'
    y_type = 'discrete'

    ## constructor
    def __init__(self, sigma=1.0):
        self.set_sigma(sigma)
        self.log2pi = np.log(2.0*np.pi)

    ## set_hyper
    # Sets the hyperparameters for the probit
    # @param hyper - a sequence with [sigma]
    def set_hyper(self, hyper):
        self.set_sigma(hyper[0])

    ## get_hyper
    # Gets a numpy array of hyperparameters for the probit
    def get_hyper(self):
        return np.array([self.sigma])

    ## set_sigma
    # Sets the sigma value, and also sets up a couple of useful constants to go with it.
    # @param sigma - the sigma to use
    def set_sigma(self, sigma):
        self.sigma = sigma
        self._isqrt2sig = 1.0 / (self.sigma * np.sqrt(2.0))
        self._i2var =  self._isqrt2sig**2


    ## print_hyperparameters
    # prints the hyperparameter of the probit
    def print_hyperparameters(self):
        print("Probit relative, Gaussian noise on latent. Sigma: {0:0.2f}".format(self.sigma))

    ## z_k
    # returns the probit for the given probit class (the difference in the
    # latent space)
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the input estimates in the latent space
    #
    # @return the vector of z_k values
    def z_k(self, y, F):
        return self._isqrt2sig * y[:,0] * (F[y[:,2]] - F[y[:,1]])

    ## derv_discrete_loglike
    # Calculates the first derivative of log likelihood.
    # Appendix A.1.1.1.1
    # Assumes (f(vk), f(uk))
    #
    # xi, yk, vk, are indicies of the likelihood
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @return - the values for the u and v of y numpy (n,2) [[]]
    def derv_log_likelyhood(self, y, F):
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(self.z_k(y, F))
        derv_ll_pairs = y[:,0] * pdf_cdf_ratio * self._isqrt2sig
        derv_ll = np.zeros(len(F))


        derv_ll = add_up_vec(y[:,1], -derv_ll_pairs, derv_ll)
        derv_ll = add_up_vec(y[:,2], +derv_ll_pairs, derv_ll)

        # for i in range(len(y)):
        #     derv_ll[y[i,1]] -= derv_ll_pairs[i]
        #     derv_ll[y[i,2]] += derv_ll_pairs[i]

        return derv_ll

    ## grad_hyper
    # Calculates the gradient of p(y|F) given the parameters of the probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return numpy array (gradient of probit with respect to hyper parameters)
    def grad_hyper(self, y, F):
        zk = self.z_k(y, F)
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(zk)

        dP_dSigma = -np.sum(zk * pdf_cdf_ratio) / self.sigma

        return np.array([dP_dSigma])

    ## calc_W
    # caclulate the W matrix
    # Calculates the second derivative of discrete log likelihood.
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    #  d / (dF(xi)dF(xj)) ln(p(dk|F(u), F(v)))
    #
    # Appendix A.1.1.2
    # Assumes (f(vk), f(uk))
    # xi, yk, vk, are indicies of the likelihood
    # @param F - the vector of f (estimated training sample outputs)
    def calc_W(self, y, F):
        z = self.z_k(y, F)
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(z)

        paren_pairs = np.where(np.logical_and(z < 0, np.isinf(pdf_cdf_ratio)), 0, \
                        (z * pdf_cdf_ratio) + pdf_cdf_ratio2)
        d2_ll_pairs = -(y[:,0]*y[:,0])*paren_pairs*self._i2var

        W = np.zeros((len(F), len(F)))

        # vectorized versions of summation
        idx1 = np.array([y[:,1], y[:,1]]).T
        idx2 = np.array([y[:,1], y[:,2]]).T
        idx3 = np.array([y[:,2], y[:,1]]).T
        idx4 = np.array([y[:,2], y[:,2]]).T

        W = add_up_mat(idx1, -d2_ll_pairs, W)
        W = add_up_mat(idx2,  d2_ll_pairs, W)
        W = add_up_mat(idx3,  d2_ll_pairs, W)
        W = add_up_mat(idx4, -d2_ll_pairs, W)

        return W

    ## calc_W_dF
    # Calculate the third derivative of the W matrix.
    # d ln(p(dk|F(u), F(v))) / d f_i, f_j, f_k
    # This returns a 3d matrix of (N x N x N) where N is the length of the F vector.
    # Equation (65)
    #
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @reutrn 3d matrix
    def calc_W_dF(self, y, F):
        z = self.z_k(y, F)
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(z)

        paren_pairs = np.where(np.logical_and(z < 0, np.isinf(pdf_cdf_ratio)), 0, \
                        pdf_cdf_ratio - z*z*pdf_cdf_ratio - 3*z*pdf_cdf_ratio2 - 2 * pdf_cdf_ratio2 * std_norm_pdf(z))
        paren_pairs *= y[:,0]*y[:,0]*y[:,0]
        paren_pairs *= -1 / (2 * np.sqrt(2) * self.sigma * self.sigma * self.sigma)


        N = len(F)
        dW = np.zeros((N, N, N))
        dW = add_up_W_partial(y, paren_pairs, dW)

        return dW

    ## calc_W_dHyper
    # Calculate the derivative of the W matrix with respect to hyper parameters.
    # dW / dSigma
    # This returns a 3d matrix of (N x N x N) where N is the length of the F vector.
    # Equation (70)
    #
    # @param y - the label for the given probit (dk, u, v) (must be a numpy array)
    # @param F - the vector of F (estimated training sample outputs)
    #
    # @reutrn 2d matrix
    def calc_W_dHyper(self, y, F):
        z = self.z_k(y, F)
        pdf_cdf_ratio, pdf_cdf_ratio2 = calc_pdf_cdf_ratio(z)
        pdf = std_norm_pdf(z)
        cdf = std_norm_cdf(z)

        paren_pairs = np.where(np.logical_and(z < 0, np.isinf(pdf_cdf_ratio)), 0, \
                        (z * pdf_cdf_ratio) + pdf_cdf_ratio2)
        term1 = (y[:,0] * y[:,0] / (self.sigma * self.sigma * self.sigma)) * paren_pairs
        term2a = (1 / (2 * self.sigma * self.sigma * self.sigma)) * y[:,0] * y[:,0] * z * pdf / (cdf * cdf * cdf)
        term2b = -(cdf*cdf) + z*z*cdf*cdf + 3*z*pdf*cdf + 2 * pdf*pdf

        dw_pairs = term1 - term2a*term2b

        dW = np.zeros((len(F), len(F)))

        # vectorized versions of summation
        idx1 = np.array([y[:,1], y[:,1]]).T
        idx2 = np.array([y[:,1], y[:,2]]).T
        idx3 = np.array([y[:,2], y[:,1]]).T
        idx4 = np.array([y[:,2], y[:,2]]).T

        dW = add_up_mat(idx1, -dw_pairs, dW)
        dW = add_up_mat(idx2,  dw_pairs, dW)
        dW = add_up_mat(idx3,  dw_pairs, dW)
        dW = add_up_mat(idx4, -dw_pairs, dW)

        return dW[np.newaxis, :, :]


    ## derivatives
    # Calculates the derivatives of the probit with the given input data
    # @param y - the given set of labels for the probit
    #              this is given as a list of [(dk, u, v), ...]
    # @param F - the input data samples
    #
    # @return - W, dpy_df, py
    #       W - is the second order derivative of the probit with respect to F
    #       dpy_df - the derivative of log P(y|x,theta) with respect to F
    #       py - log P(y|x,theta) for the given probit
    def derivatives(self, y, F):
        py = self.log_likelihood(y, F)
        dpy_df = self.derv_log_likelyhood(y, F)
        W = self.calc_W(y, F)

        return W, dpy_df, py


    ## likelihood
    # Returns the liklihood function for the given probit
    # @param y - the given set of labels for the probit
    # @param F - the input data samples
    #
    # @return P(y|F)
    def likelihood(self, y, F):
        z = self.z_k(y, F)
        return std_norm_cdf(z)








try:
    import numba

    ## add_up_mat
    # add the values in v to the M matrix indexed by the indicies matrix
    # @param indicies - (n x k) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the matrix to add up
    @numba.jit
    def add_up_mat(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i,0], indicies[i,1]] += v_i

        return M

    ## add_up_vec
    # add the values in v to the M vector indexed by the indicies matrix
    # @param indicies - (n,) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the vector to add up
    @numba.jit
    def add_up_vec(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i]] += v_i

        return M

    ## add_up_W_partial
    # add and place the y inidicies into the appropriate locations.
    # @param y - numpy array of labels
    # @param v - the values going with the labels (paren pairs)
    # @param dW - the derivative of W np array to add the value pairs to.
    @numba.jit
    def add_up_W_partial(y, v, dW):
        for i, y_i in enumerate(y):
            dW[y_i[1], y_i[1], y_i[1]] += v[i]
            dW[y_i[2], y_i[1], y_i[1]] += -v[i]

            dW[y_i[1], y_i[1], y_i[2]] += -v[i]
            dW[y_i[2], y_i[1], y_i[2]] += v[i]

            dW[y_i[1], y_i[2], y_i[1]] += -v[i]
            dW[y_i[2], y_i[2], y_i[1]] += v[i]

            dW[y_i[1], y_i[2], y_i[2]] += v[i]
            dW[y_i[2], y_i[2], y_i[2]] += -v[i]

        return dW
except ImportError:
    print('Failed to import numba, add_up_mat and add_up_vec will be slower')

    ## add_up_mat
    # add the values in v to the M matrix indexed by the indicies matrix
    # @param indicies - (n x k) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the matrix to add up
    def add_up_mat(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i,0], indicies[i,1]] += v_i

        return M

    ## add_up_vec
    # add the values in v to the M vector indexed by the indicies matrix
    # @param indicies - (n,) the list of indicies for the v to add to the matrix
    # @param v - the values ligned up wiht indicies
    # @param M -  [in/out] the vector to add up
    def add_up_vec(indicies, v, M):
        for i, v_i in enumerate(v):
            M[indicies[i]] += v_i

        return M

    ## add_up_W_partial
    # add and place the y inidicies into the appropriate locations.
    # @param y - numpy array of labels
    # @param v - the values going with the labels (paren pairs)
    # @param dW - the derivative of W np array to add the value pairs to.
    def add_up_W_partial(y, v, dW):
        for i, y_i in enumerate(y):
            dW[y_i[1], y_i[1], y_i[1]] += v[i]
            dW[y_i[2], y_i[1], y_i[1]] += -v[i]

            dW[y_i[1], y_i[1], y_i[2]] += -v[i]
            dW[y_i[2], y_i[1], y_i[2]] += v[i]

            dW[y_i[1], y_i[2], y_i[1]] += -v[i]
            dW[y_i[2], y_i[2], y_i[1]] += v[i]

            dW[y_i[1], y_i[2], y_i[2]] += v[i]
            dW[y_i[2], y_i[2], y_i[2]] += -v[i]

        return dW






















#
