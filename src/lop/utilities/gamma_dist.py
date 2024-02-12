# Copyright 2024 Ian Rankin
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

# gamma_dist.py
# Written Ian Rankin - Febuary 2024
#
# An implementation of gamma distributions.
# This is used instead of scipy stats because the derivative is required.
# Also allows both k and theta parameter directly to be specified

import numpy as np
from scipy.special import gamma



## pdf_gamma
# returns the pdf of the gamma distribution (k, theta derivation)
# https://en.wikipedia.org/wiki/Gamma_distribution
# mean = k*theta
# @param x - the input set x input to the pdf
# @param k - the shape parameter of the gamma distribution
# @param theta - the scale parameter of the gamma distribution
def pdf_gamma(x, k, theta):
    term1 = 1 / (gamma(k) * (theta**k))
    
    return term1 * (x ** (k-1)) * np.exp(-x / theta)

## log_pdf_gamma
# returns the log of the pdf of the gamma distribution (k, theta derivation)
# https://en.wikipedia.org/wiki/Gamma_distribution
# mean = k*theta
# @param x - the input set x input to the pdf
# @param k - the shape parameter of the gamma distribution
# @param theta - the scale parameter of the gamma distribution
def log_pdf_gamma(x, k, theta):
    term1 = 1 / (gamma(k) * (theta**k))
    
    x = np.where(x < 0.000001, 0.000001, x)
    return np.log(term1) + (k-1)*np.log(x) - (x / theta)

## d_log_pdf_gamma
# returns the derivative of the log of the pdf of the gamma distribution (k, theta derivation)
# https://en.wikipedia.org/wiki/Gamma_distribution
# mean = k*theta
# @param x - the input set x input to the pdf
# @param k - the shape parameter of the gamma distribution
# @param theta - the scale parameter of the gamma distribution
def d_log_pdf_gamma(x, k, theta):
    x = np.where(x < 0.000001, 0.000001, x)
    return ((k - 1) / x) - (1 / theta)


