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

# pref_gp_abs_bound.py
# Written Ian Rankin - December 2023
#
# An example usage of a simple pref with abs bound probit.
# Abs bound requires continous input values between 0,1


import numpy as np
import matplotlib.pyplot as plt
import argparse

import lop


def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def main():
    parser = argparse.ArgumentParser(description='beta distribution plotter')
    parser.add_argument('-v', type=float, default=80, help='the precision variable on the distribution')
    parser.add_argument('--abs_sigma', type=float, default=1.0, help='Enter sigma parameter of the mean link of the beta distribution (scale parameter)')
    parser.add_argument('--pair_sigma', type=float, default=1.0, help='Enter sigma parameter of the pairwise the kernel')
    parser.add_argument('--rbf_sigma', type=float, default=1.0, help='Enter sigma parameter of the rbf kernel')
    parser.add_argument('--rbf_l', type=float, default=0.7, help='Enter lengthscale of rbf kernel')
    args = parser.parse_args()

    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(args.rbf_sigma, args.rbf_l, sigma_noise=0.000001))
    gp.probits[0].set_sigma(args.pair_sigma)
    gp.probits[2].set_v(args.v)
    gp.probits[2].set_sigma(args.abs_sigma)
    
    X_train = np.array([0.3, 1.0, 1.8, 6.9])
    y_train = lop.normalize_0_1(f_sin(X_train), 0.05)

    gp.add(X_train, y_train, type='abs')

    X_train = np.array([0,2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2)

    gp.add(X_train, pairs)

    gp.optimize()

    # predict output of GP
    X = np.arange(0.0, 9.0, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    

    print(gp.n_loops)

    # Plotting output for easy viewing
    plt.plot(X, mu)
    sigma_to_plot = 1

    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    Y_actual = f_sin(X)
    Y_max = np.linalg.norm(Y_actual, ord=np.inf)
    Y_actual = Y_actual / Y_max
    plt.plot(X, Y_actual)
    plt.scatter(gp.X_train, gp.F)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted function with predicted F', 'Real function'])
    gp.plot_preference(head_size=25)
    plt.show()


if __name__ == '__main__':
    main()
