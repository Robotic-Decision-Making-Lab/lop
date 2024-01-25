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

# sample_prior.py
# Written Ian Rankin - January 2024
#
# An example to show sampling a function with an RBF kernel with no posterior

import numpy as np
import matplotlib.pyplot as plt

import lop

def main():
    # setup the variables to function, and the mean of the function
    x = np.arange(0,10,0.01)
    mu = np.zeros(len(x))

    # define the GP and calculate the covariance of the noise function
    k = lop.RBF_kern(sigma=1.0, l=1.0)
    cov = k.cov(x, x)

    # sample the function
    f = np.random.multivariate_normal(mu, cov)

    sigma_to_plot = 1
    std = np.sqrt(np.diagonal(cov))
    plt.gca().fill_between(x, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    plt.plot(x, f)
    plt.xlabel('x')
    plt.ylabel('function')
    plt.title('GP sample with only the prior')
    plt.legend(['sampled function', 'standard deviation of random variable'])
    plt.show()

if __name__ == '__main__':
    main()
