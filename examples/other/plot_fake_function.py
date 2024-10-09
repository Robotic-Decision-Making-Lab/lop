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

# plot_fake_function.py
# Written Ian Rankin - January 2024
#
# An example usage of plotting one of the random fake functions
# This allows testing to see what the functions look like.

import numpy as np
import matplotlib.pyplot as plt
import argparse
import lop

import pdb

def main():
    parser = argparse.ArgumentParser(description='Fake function plotter')
    parser.add_argument('-f', type=str, default='logistic', help='Enter the type of function [linear squared logistic sin_exp max min squared_min_max mix_gauss int_gauss]')
    parser.add_argument('-d', type=int, default=1, help='Enter the dimmensionality of the fake function (1 or 2) for plotting')
    args = parser.parse_args()

    dim = args.d

    fc = None
    if args.f == 'linear':
        fc = lop.FakeLinear(dim)
    elif args.f == 'squared':
        fc = lop.FakeSquared(dim)
    elif args.f == 'logistic':
        fc = lop.FakeLogistic(dim)
    elif args.f == 'sin_exp':
        fc = lop.FakeSinExp(dim)
    elif args.f == 'max':
        fc = lop.FakeWeightedMax(dim)
    elif args.f == 'min':
        fc = lop.FakeWeightedMin(dim)
    elif args.f == 'squared_min_max':
        fc = lop.FakeSquaredMinMax(dim)
    elif args.f == 'mix_gauss':
        fc = lop.FakeMixtureGaussian(dim)
    elif args.f == 'int_gauss':
        f_to_int = lop.FakeMixtureGaussian(dim)
        fc = lop.FakeIntegrate(dim, f_to_int)
    else:
        print('Unknown function: ' + str(args.f))
        return

    if dim == 1:
        x = np.arange(0,5,0.01)
        y = fc(x)

        plt.plot(x,y)
        plt.xlabel('x input values')
        plt.ylabel('y output values')
        plt.show()    
    elif dim == 2:
        grid = np.arange(0,2,0.01)
        xv, yv = np.meshgrid(grid, grid)
        x_l = xv.reshape(-1)
        y_l = yv.reshape(-1)
        pts = np.append(x_l[:,np.newaxis], y_l[:,np.newaxis],axis=1)
        z = fc(pts)

        z_plt = z.reshape(xv.shape)
        #print(z_plt)

        fig = plt.figure()


        cont = plt.contourf(xv, yv, z_plt, levels=300)
        plt.contour(xv,yv, z_plt)
        fig.colorbar(cont, label='output values')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    else:
        print("I can't plot that dimmension")



if __name__ == '__main__':
    main()
