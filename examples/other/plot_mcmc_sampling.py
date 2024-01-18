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

# plot_mcmc_sampling.py
# Written Ian Rankin - January 2024
#
# A simple example of mcmc sampling to show the sampling is performing as expected

import numpy as np
import matplotlib.pyplot as plt

import lop

def gaussian_liklihood(x):
    return np.log(np.exp(-np.dot(x, x) / 2) / np.sqrt(2 * np.pi))

def main():
    samples = lop.metropolis_hastings(gaussian_liklihood, 500, dim=2)

    plt.plot(np.arange(0,len(samples)), samples)
    plt.figure()
    plt.hist(samples)
    plt.show()

if __name__== '__main__':
    main()