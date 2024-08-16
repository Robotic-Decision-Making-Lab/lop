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

# plot_beta_prior.py
# Written Ian Rankin - August 2024
#
# An example usage that plots the beta distribution of the prior given different input F
# and parameter v and sigma values.

import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm, beta
import lop


def main():
    parser = argparse.ArgumentParser(description='beta distribution plotter')
    parser.add_argument('-F', type=float, default=1.0, help='Enter F value to plot')
    parser.add_argument('-v', type=float, default=80, help='the precision variable on the distribution')
    parser.add_argument('--sigma', type=float, default=1.0, help='Enter sigma parameter of the mean link of the beta distribution (scale parameter)')
    args = parser.parse_args()


    q = np.arange(0.0, 1.0, 0.001)

    F = np.array([args.F])
    sigma = args.sigma
    v = args.v

    abs_probit = lop.AbsBoundProbit(sigma=sigma, v=v, eps=0.0)

    aa, bb = abs_probit.get_alpha_beta(F)

    print('aa: ' + str(aa))
    print('bb: ' + str(bb))



    y = beta.pdf(q, aa, bb)

    print(y[int(y.shape[0]/2)])

    plt.plot(q, y)
    #plt.plot(q, y_other)

    plt.title('Probit PDF over possible query values')
    plt.xlabel('q, for v='+str(v)+', sigma='+str(sigma)+', F='+str(F))
    plt.ylabel('p(q|F)')
    plt.legend(['pdf'])
    plt.show()



if __name__ == '__main__':
    main()
