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
import argparse
import sys

import lop

import pdb




def plot_data(gp):
    # Create test output of model
    # predict output of GP
    X = np.arange(-0.5, 10, 0.1)
    mu, sigma = gp.predict(X)
    std = np.sqrt(sigma)


    print('Sigma = '+str(gp.probits[0].sigma))


    # Plotting output for easy viewing
    plt.plot(X, mu)
    sigma_to_plot = 1

    plt.gca().fill_between(X, mu-(sigma_to_plot*std), mu+(sigma_to_plot*std), color='#dddddd')

    gp.plot_preference()
    plt.scatter(gp.X_train, gp.F)

    plt.title('Gaussian Process estimate (1 sigma)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['Predicted function with predicted F'])

possible_models = ['gp', 'linear']
possible_selectors = ['UCB', 'SGV_UCB', 'RANDOM', 'ABS_BAYES_PROBIT', 'ACQ_RHO', 'ACQ_EPIC', 'ACQ_LL', 'ACQ_SPEAR','SW_BAYES_PROBIT', 'SW_ACQ_RHO', 'SW_ACQ_EPIC', 'SW_ACQ_LL', 'SW_ACQ_SPEAR']
possible_prior = ['bimodal', 'empty', 'a', 'b']

def main():
    parser = argparse.ArgumentParser(description='bimodal example with different models and active learners')
    parser.add_argument('--selector', type=str, default='BAYES_INFO_GAIN', help='Set the selectors to use options '+str(possible_selectors))
    parser.add_argument('--model', type=str, default='gp', help='Set the model to '+str(possible_models))
    parser.add_argument('-v', type=float, default=80.0, help='abs probit v parameter default=80.0')
    parser.add_argument('--sigma_abs', type=float, default=1.0, help='abs probit sigma parameter default=1.0')
    parser.add_argument('--sigma_pair', type=float, default=1.0, help='abs probit sigma parameter default=1.0')
    parser.add_argument('--prior', type=str, default='bimodal', help='Selects prior given to model between '+str(possible_prior))
    args = parser.parse_args()

    if args.selector not in possible_selectors:
        print('Selector should be one of these '+str(possible_selectors)+' not ' + str(args.selector))
        sys.exit(0)
    if args.model not in possible_models:
        print('model should be one of these '+str(possible_models)+' not ' + str(args.model))
        sys.exit(0)
    if args.prior not in possible_prior:
        print('prior should be one of these '+str(possible_prior)+' not ' + str(args.prior))
        sys.exit(0)

    # Create active learner
    al = None
    if args.selector == 'UCB':
        al = lop.UCBLearner()
    elif args.selector == 'SGV_UCB':
        al = lop.GV_UCBLearner()
    elif args.selector == 'RANDOM':
        al = lop.RandomLearner()
    elif args.selector == 'ABS_BAYES_PROBIT':
        al = lop.AbsBayesInfo(M=200)
    elif args.selector == 'ACQ_RHO':
        al = lop.AbsAcquisition(M=200, alignment_f='rho')
    elif args.selector == 'ACQ_LL':
        al = lop.AbsAcquisition(M=200, alignment_f='loglikelihood')   
    elif args.selector == 'ACQ_EPIC':
        al = lop.AbsAcquisition(M=200, alignment_f='epic')
    elif args.selector == 'ACQ_SPEAR':
        al = lop.AbsAcquisition(M=200, alignment_f='spearman')
    elif args.selector == 'SW_ACQ_RHO':
        abs = lop.AbsAcquisition(M=200, alignment_f='rho')
        pair = lop.AcquisitionSelection(M=200, alignment_f='rho')
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs)
    elif args.selector == 'SW_ACQ_LL':
        abs = lop.AbsAcquisition(M=200, alignment_f='loglikelihood')
        pair = lop.AcquisitionSelection(M=200, alignment_f='loglikelihood')
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs)
    elif args.selector == 'SW_ACQ_EPIC':
        abs = lop.AbsAcquisition(M=200, alignment_f='epic')
        pair = lop.AcquisitionSelection(M=200, alignment_f='epic')
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs)
    elif args.selector == 'SW_ACQ_SPEAR':
        abs = lop.AbsAcquisition(M=200, alignment_f='spearman')
        pair = lop.AcquisitionSelection(M=200, alignment_f='spearman')
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs)
    elif args.selector == 'SW_BAYES_PROBIT':
        abs = lop.AbsBayesInfo(M=200)
        pair = lop.BayesInfoGain()
        al = lop.RateChooseLearner(pairwise_l=pair, abs_l=abs)
    



    #### create model
    if args.model == 'gp':
        model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7, sigma_noise=0.000001), active_learner=al, normalize_gp=False, use_hyper_optimization=False)

    if args.model == 'linear':
        model = lop.PreferenceLinear(active_learner=al)

    model.probits[0].set_sigma(args.sigma_pair)
    model.probits[2].set_sigma(args.sigma_abs)
    model.probits[2].set_v(args.v)

    #model.add(np.array([5]), np.array([0.5]), type='abs')
    #
    print('v = ' + str(model.probits[2].v))
    print('pair sigma = ' + str(model.probits[0].sigma))
    print('abs sigma = ' + str(model.probits[2].sigma))


    if args.prior == 'bimodal':
        X_train = np.array([0,1,2,3,4,5,6,7,8,9,9.5])
        pairs = [   lop.preference(2,0),
                    lop.preference(2,1),
                    lop.preference(2,3),
                    lop.preference(2,4),
                    lop.preference(7,6),
                    lop.preference(7,5),
                    lop.preference(7,9),
                    lop.preference(8,10),
                    lop.preference(8,9)]

        model.add(X_train, pairs)
    elif args.prior == 'empty':
        pass
    elif args.prior == 'a':
        X_train = np.array([0,1,2,3,4,5,6,7,8,9,9.5])
        pairs = [   lop.preference(4,0),
                    lop.preference(3,2),
                    lop.preference(6,9),
                    lop.preference(7, 10),
                    lop.preference(1,8),
                    lop.preference(5,4)]

        model.add(X_train, pairs)
    elif args.prior == 'b':
        X_train = np.array([0,1,4,5,6,9.5])
        pairs = [   lop.preference(2,0),
                    lop.preference(1, 4),
                    lop.preference(2,3),
                    lop.preference(1, 5)]

        x_abs = np.array([3, 7, 9.2, 2.2])
        y_abs = np.array([0.4, 0.7, 0.1, 0.36])

        model.add(X_train, pairs)
        model.add(x_abs, y_abs, type='abs')

    plot_data(model)


    # carefully selected to have 2.1 and 7.5 (indicies 0 and 1) to be the highest
    # information gain points. (disambiguates which of the two peaks is higher.)
    x_canidiates = np.array([2.1, 7.5, 0.5, 4.5,5.5,9.2])

    #model.probits[0].set_sigma(0.1)
    test_pt_idxs = model.select(x_canidiates, 2)
    plt.scatter(x_canidiates[test_pt_idxs], model(x_canidiates[test_pt_idxs]))

    plt.show()

if __name__ == '__main__':
    main()

