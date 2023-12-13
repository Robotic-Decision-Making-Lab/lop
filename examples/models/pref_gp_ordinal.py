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

# pref_gp_ordinal.py
# Written Ian Rankin - December 2023
#
# An example usage of a simple preference GP with discrete ordinal data.


import numpy as np
import matplotlib.pyplot as plt

import lop



def main():
    X_train = np.array([0,1,2,3,4.2,6,7])
    ratings = np.array([5,5,2,1,2  ,3,3])


    gp = lop.PreferenceGP(lop.RBF_kern(0.5, 0.7))
    gp.add(X_train, ratings, type='ordinal')



    X = np.arange(-0.5, 8, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)

    
    sigma_to_plot = 1


    # GP scaled between 0 to 1, use scaling to match ratings output
    scale = 2.5
    inter = 2.5
    plt.plot(X, mu*scale + inter)
    #plt.scatter(X_train, gp.F)
    
    plt.gca().fill_between(X, (mu-(sigma_to_plot*std))*scale+inter, (mu+(sigma_to_plot*std))*scale+inter, color='#dddddd')
    plt.scatter(X_train, gp.F*scale + inter)
    plt.scatter(X_train, ratings, color='orange')
    

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted function', 'Uncert', 'Predicted ratings', 'Given ratings'])
    plt.show()


if __name__ == '__main__':
    main()
