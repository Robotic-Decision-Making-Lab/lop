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

# plot_gamma_distribution.py
# Written Ian Rankin - Febuary 2024
#
# An example usage of plotting the gamma distribution used in the prior for the hyperparameters
# This allows testing to see what the functions look like.

import numpy as np
import matplotlib.pyplot as plt
import argparse
import lop


def main():
    parser = argparse.ArgumentParser(description='Gamma distribution plotter')
    parser.add_argument('-k', type=float, default=10.0, help='Enter k parameter of the gamma distribution (shape parameter)) mean = k*theta')
    parser.add_argument('--theta', type=float, default=0.1, help='Enter theta parameter of the gamma distribution (scale parameter) mean = k*theta')
    parser.add_argument('--max_x', type=float, default=200.0, help='Enter the max x to plot')
    args = parser.parse_args()


    x = np.arange(0.1,args.max_x,0.1)

    if args.max_x < 10.0:
        x = np.arange(0.01, args.max_x, 0.001)

    k = args.k
    theta = args.theta

    y = lop.pdf_gamma(x, k, theta)
    y2 = lop.log_pdf_gamma(x, k, theta)
    dy_log = lop.d_log_pdf_gamma(x, k, theta)

    plt.plot(x, y)
    plt.plot(x, y2)
    plt.plot(x,dy_log)
    plt.xlabel('x')
    plt.ylabel('PDF / log PDF / d/dx log PDF')
    plt.legend(['pdf', 'log pdf', 'gradient log PDF'])
    plt.show()



if __name__ == '__main__':
    main()
