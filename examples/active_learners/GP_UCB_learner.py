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

# GP_UCB_learner.py
# Written Ian Rankin - December 2023
#
# An example usage of standard GP with a UCB active learning algorithm
# This is done using a 1D function.


import numpy as np
import matplotlib.pyplot as plt

import lop


# the function to approximate
def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def main():
    al = lop.UCBLearner()
    model = lop.GP(lop.RBF_kern(0.5,1.0), active_learner=al)


    # Generate active learning point and add it to the model
    for i in range(10):
        # generate random test set to select test point from
        x_canidiates = np.random.random(20)*10

        test_pt_idxs = model.select(x_canidiates, 2)


        x_train = x_canidiates[test_pt_idxs]
        y_train = f_sin(x_train)

        model.add(x_train, y_train)

    # Create test output of model
    x_test = np.arange(0,10,0.01)
    y_test = f_sin(x_test)
    y_pred,y_sigma = model.predict(x_test)
    std = np.sqrt(y_sigma)

    # Pring output of model with uncertianty
    sigma_to_plot = 1.96
    plt.plot(x_test, y_test)
    plt.plot(x_test, y_pred)
    plt.scatter(model.X_train, model.y_train)
    plt.gca().fill_between(x_test, y_pred-(sigma_to_plot*std), y_pred+(sigma_to_plot*std), color='#dddddd')
    plt.xlabel('input values')
    plt.ylabel('GP output')
    plt.legend(['Real function', 'Predicted function', 'Active learning points', '95% condidence region'])
    plt.show()

if __name__ == '__main__':
    main()

