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

# pref_gp.py
# Written Ian Rankin - December 2023
#
# An example usage of a simple user gaussian process on a 1D example
# generates pairwise training data


import numpy as np
import matplotlib.pyplot as plt

import lop

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def main():
    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))
    gp.add(X_train, pairs)
    gp.optimize(optimize_hyperparameter=False)

    # predict output of GP
    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)


    print('Sigma = '+str(gp.probits[0].sigma))


    # Plotting output for easy viewing
    plt.plot(X, mu)
    sigma_to_plot = 1

    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    Y_actual = f_sin(X)
    Y_max = np.linalg.norm(Y_actual, ord=np.inf)
    Y_actual = Y_actual / Y_max
    plt.plot(X, Y_actual)
    plt.scatter(X_train, gp.F)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted function with predicted F', 'Real function'])
    plt.show()


if __name__ == '__main__':
    main()
