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

# hyperparameter_gp.py
# Written Ian Rankin - Febuary 2024
#
# An example usage of hyperparameter optimization for the preference GP.


import numpy as np
import argparse
import matplotlib.pyplot as plt

import lop

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def main():
    parser = argparse.ArgumentParser(description='Pa')
    parser.add_argument('-i', type=str, default='full', help='Enter the type of pairs [full weird]')
    args = parser.parse_args()

    # Create preference gp and optimize given training data
    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7), normalize_gp=False, normalize_positive=False)
    gp.probits[0].set_sigma(0.2)

    if args.i == 'full':
        X_train = np.array([0,1,2,3,4.2,6,7])
        pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
                lop.generate_fake_pairs(X_train, f_sin, 1) + \
                lop.generate_fake_pairs(X_train, f_sin, 2) + \
                lop.generate_fake_pairs(X_train, f_sin, 3) + \
                lop.generate_fake_pairs(X_train, f_sin, 4)
        
        gp.add(X_train, pairs)
    elif args.i == 'weird':
        X_train = np.array([0,1.9999,2,2.0001,4.2,6,7, 4.7])
        y_train = f_sin(X_train)

        pairs = lop.gen_pairs_from_idx(np.argmax(y_train[0:3]), list(range(len(y_train[0:3]))))
        
        pairs2 = lop.gen_pairs_from_idx(np.argmax(y_train[3:5]), list(range(len(y_train[3:5]))))
        pairs2 = [(p[0], p[1]+3, p[2]+3) for p in pairs2]
        pairs += pairs2
        pairs2= lop.gen_pairs_from_idx(np.argmax(y_train[5:]), list(range(len(y_train[5:]))))
        pairs2 = [(p[0], p[1]+5, p[2]+5) for p in pairs2]
        pairs += pairs2

        gp.add(X_train, pairs)

    elif args.i == 'abs':
        X_train = np.array([0.0, 1.0, 1.8, 3.0, 5.6, 6.9])
        y_train = lop.normalize_0_1(f_sin(X_train), 0.05)

        gp.add(X_train, y_train, type='abs')

    elif args.i == 'weird_abs':
        X_train = np.array([0,1.9999,2,2.0001,4.2,6,7, 4.7])
        y_train = f_sin(X_train)

        pairs = lop.gen_pairs_from_idx(np.argmax(y_train[0:3]), list(range(len(y_train[0:3]))))
        
        pairs2 = lop.gen_pairs_from_idx(np.argmax(y_train[3:5]), list(range(len(y_train[3:5]))))
        pairs2 = [(p[0], p[1]+3, p[2]+3) for p in pairs2]
        pairs += pairs2
        pairs2= lop.gen_pairs_from_idx(np.argmax(y_train[5:]), list(range(len(y_train[5:]))))
        pairs2 = [(p[0], p[1]+5, p[2]+5) for p in pairs2]
        pairs += pairs2

        gp.add(X_train, pairs)

        X_train = np.array([5.0])
        y_train = np.array([0.5])

        gp.add(X_train, y_train, type='abs')
    elif args.i == 'full_abs':
        X_train = np.array([0,1,2,3,4.2,6,7])
        pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
                lop.generate_fake_pairs(X_train, f_sin, 1) + \
                lop.generate_fake_pairs(X_train, f_sin, 2) + \
                lop.generate_fake_pairs(X_train, f_sin, 3) + \
                lop.generate_fake_pairs(X_train, f_sin, 4)
        
        gp.add(X_train, pairs)

        X_train = np.array([5])
        y_train = np.array([0.5])
        gp.add(X_train, y_train, type='abs')    


    
    gp.optimize(optimize_hyperparameter=True)

    # predict output of GP
    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)



    # Plotting output for easy viewing
    plt.figure()
    ax = plt.gca()
    ax.plot(X, mu)
    sigma_to_plot = 1

    ax.fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')
    Y_actual = f_sin(X)
    Y_max = np.linalg.norm(Y_actual, ord=np.inf)
    Y_actual = Y_actual / Y_max
    ax.plot(X, Y_actual)
    ax.legend(['Predicted function with predicted F', 'Real function'])
    gp.plot_preference(head_width=0.1, ax=ax)
    
    ax.scatter(gp.X_train, gp.F)
    if args.i == 'abs':
        ax.scatter(X_train, y_train)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')

    plt.show()


if __name__ == '__main__':
    main()
