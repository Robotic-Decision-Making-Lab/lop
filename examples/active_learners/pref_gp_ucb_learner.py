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


def plot_data(model, new_selections=None):
    # Create test output of model
    x_test = np.arange(0,10,0.005)
    y_test = f_sin(x_test)
    y_pred,y_sigma = model.predict(x_test)
    std = np.sqrt(y_sigma)

    # Plot output of model with uncertianty
    sigma_to_plot = 1.96

    plt.clf()
    ax = plt.gca()
    ax.plot(x_test, y_test, zorder=5)
    ax.plot(x_test, y_pred, zorder=5)

    if hasattr(model, "F"):
        ax.scatter(model.X_train, model.F, zorder=10)

    
    ax.fill_between(x_test, y_pred-(sigma_to_plot*std), y_pred+(sigma_to_plot*std), color='#dddddd', zorder=1)
    if new_selections is not None:
        print('Show scatter?')
        print(new_selections)
        print(model.F[-len(new_selections):])
        ax.scatter(new_selections, model.F[-len(new_selections):], color='orange', zorder=20)
    plt.xlabel('input values')
    plt.ylabel('GP output')
    plt.legend(['Real function', 'Predicted function', 'Active learning points', '95% condidence region'])
    model.plot_preference(ax)

def main():
    #al = lop.GV_UCBLearner()
    al = lop.RandomLearner()
    #al = lop.UCBLearner()
    model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7), active_learner=al, normalize_gp=False, use_hyper_optimization=True)
    model.probits[0].set_sigma(0.5)

    fig = plt.figure()
    writer = FFMpegWriter(fps=1)

    with writer.saving(fig, "pref_gp.gif", 100):

        plot_data(model)
        writer.grab_frame()

        # Generate active learning point and add it to the model
        for i in range(10):
            # generate random test set to select test point from
            x_canidiates = np.random.random(20)*10

            test_pt_idxs = model.select(x_canidiates, 2)


            x_train = x_canidiates[test_pt_idxs]
            y_train = f_sin(x_train)
            y_pairs = lop.gen_pairs_from_idx(np.argmax(y_train), list(range(len(y_train))))
            # y_pairs = lop.generate_fake_pairs(x_train, f_sin, 0) + \
            #             lop.generate_fake_pairs(x_train, f_sin, 1) + \
            #             lop.generate_fake_pairs(x_train, f_sin, 2)

            model.add(x_train, y_pairs)

            plot_data(model, x_train)
            writer.grab_frame()
        
        # plot_data(model)
        # writer.grab_frame()
    
    plt.show()

if __name__ == '__main__':
    main()

