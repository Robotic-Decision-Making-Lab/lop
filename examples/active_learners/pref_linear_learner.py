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

# GP_UCB_learner.py
# Written Ian Rankin - January 2024
#
# An example usage of preference GP with a UCB active learning algorithm
# This is done using a 1D function.


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FFMpegWriter

import lop


# the function to approximate
def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def plot_data(model, new_selections=None, w=None):
    # Create test output of model
    grid = np.arange(0,10,0.1)
    xv, yv = np.meshgrid(grid, grid)
    x_l = xv.reshape(-1)
    y_l = yv.reshape(-1)
    x_test = np.append(x_l[:,np.newaxis], y_l[:,np.newaxis],axis=1)
    #y_test = f_sin(x_test)
    y_pred,y_sigma = model.predict(x_test)
    #std = np.sqrt(y_sigma)


    plt.clf()

    plt.contour(xv,yv, y_pred.reshape(xv.shape))

    if hasattr(model, "X_train") and model.X_train is not None:
        plt.scatter(model.X_train[:,0], model.X_train[:,1], zorder=10)

    if new_selections is not None:
        plt.scatter(new_selections[:,0], new_selections[:,1], color='orange', zorder=20)

    if w is not None:
        plt.arrow(0,0,w[0]*2,w[1]*2)

def main():
    al = lop.GV_UCBLearner()
    al = lop.RandomLearner()
    model = lop.PreferenceLinear(active_learner=al)
    model.probits[0].set_sigma(0.5)

    fig = plt.figure()
    writer = FFMpegWriter(fps=1)

    fc = lop.FakeLinear(2)

    with writer.saving(fig, "pref_linear.gif", 100):

        plot_data(model, w=fc.w)
        writer.grab_frame()

        # Generate active learning point and add it to the model
        for i in range(10):
            # generate random test set to select test point from
            x_canidiates = np.random.random((20, 2))*10

            test_pt_idxs = model.select(x_canidiates, 2)


            x_train = x_canidiates[test_pt_idxs]
            y_train = fc(x_train)
            y_pairs = lop.gen_pairs_from_idx(np.argmax(y_train), list(range(len(y_train))))
            # y_pairs = lop.generate_fake_pairs(x_train, f_sin, 0) + \
            #             lop.generate_fake_pairs(x_train, f_sin, 1) + \
            #             lop.generate_fake_pairs(x_train, f_sin, 2)

            model.add(x_train, y_pairs)

            plot_data(model, x_train, fc.w)
            writer.grab_frame()
        
        # plot_data(model)
        # writer.grab_frame()
    
    plt.show()

if __name__ == '__main__':
    main()

