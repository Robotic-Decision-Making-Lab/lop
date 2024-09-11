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
import argparse

import lop

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def plot_data(model, new_selections=None, w=None):
    # Create test output of model
    grid = np.arange(0,2.0,0.02)
    xv, yv = np.meshgrid(grid, grid)
    x_l = xv.reshape(-1)
    y_l = yv.reshape(-1)
    x_test = np.append(x_l[:,np.newaxis], y_l[:,np.newaxis],axis=1)
    #y_test = f_sin(x_test)
    y_pred,y_sigma = model.predict(x_test)
    #std = np.sqrt(y_sigma)


    plt.clf()
    ax = plt.gca()
    ax.set_aspect(1.0)

    ax.contour(xv,yv, y_pred.reshape(xv.shape))

    if hasattr(model, "X_train") and model.X_train is not None:
        ax.scatter(model.X_train[:,0], model.X_train[:,1], zorder=10)

    if new_selections is not None:
        ax.scatter(new_selections[:,0], new_selections[:,1], color='orange', zorder=20)

    model.plot_preference(ax)


def main():
    parser = argparse.ArgumentParser(description='pref gp')
    parser.add_argument('-v', type=float, default=80, help='the precision variable on the distribution')
    parser.add_argument('--abs_sigma', type=float, default=1.0, help='Enter sigma parameter of the mean link of the beta distribution (scale parameter)')
    parser.add_argument('--pair_sigma', type=float, default=1.0, help='Enter sigma parameter of the pairwise the kernel')
    args = parser.parse_args()
    
    
    X_train = np.array([[1.2, 0.1], [0.3, 1.1], [0.7, 0.7], [1.4, 1.2]])
    pairs = [lop.preference(3, 2)] + \
            [lop.preference(0, 1)] + \
            [lop.preference(0,2)]


    # Create preference gp and optimize given training data
    model = lop.PreferenceLinear()
    model.probits[0].set_sigma(args.pair_sigma)
    model.probits[2].set_v(args.v)
    model.probits[2].set_sigma(args.abs_sigma)
    
    #model.add(X_train, pairs)

    X_train = np.array([[0.6, 0.92], [1.1, 0.5], [0.8, 0.5]])
    y_train = np.array([0.8, 0.86, 0.3])

    model.add(X_train, y_train, type='abs')

    model.optimize(optimize_hyperparameter=False)

    plot_data(model)
    plt.show()

if __name__ == '__main__':
    main()
