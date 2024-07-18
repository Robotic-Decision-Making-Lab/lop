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
possible_selectors = ['UCB', 'SGV_UCB', 'RANDOM', 'MUTUAL_INFO', 'MUTUAL_INFO_PERF', 'BAYES_INFO_GAIN', "PROB_LEANER", 'ACQ_RHO', 'ACQ_EPIC', 'ACQ_LL', 'ACQ_SPEAR']

def main():
    parser = argparse.ArgumentParser(description='bimodal example with different models and active learners')
    parser.add_argument('--selector', type=str, default='BAYES_INFO_GAIN', help='Set the selectors to use options '+str(possible_selectors))
    parser.add_argument('--model', type=str, default='gp', help='Set the model to '+str(possible_models))
    args = parser.parse_args()

    if args.selector not in possible_selectors:
        print('Selector should be one of these '+str(possible_selectors)+' not ' + str(args.selector))
        sys.exit(0)
    if args.model not in possible_models:
        print('model should be one of these '+str(possible_models)+' not ' + str(args.model))
        sys.exit(0)

    # Create active learner
    al = None
    if args.selector == 'UCB':
        al = lop.UCBLearner()
    elif args.selector == 'SGV_UCB':
        al = lop.GV_UCBLearner()
    elif args.selector == 'MUTUAL_INFO':
        al = lop.MutualInfoLearner()
    elif args.selector == 'MUTUAL_INFO':
        al = lop.MutualInfoLearner()
    elif args.selector == 'MUTUAL_INFO_PERF':
        al = lop.MutualInfoLearner()
    elif args.selector == 'RANDOM':
        al = lop.RandomLearner()
    elif args.selector == 'BAYES_INFO_GAIN':
        al = lop.BayesInfoGain2()
    elif args.selector == 'PROB_LEARNER':
        al = lop.ProbabilityLearner()
    elif args.selector == 'ACQ_RHO':
        al = lop.AcquisitionSelection(M=400, alignment_f='rho')
    elif args.selector == 'ACQ_LL':
        al = lop.AcquisitionSelection(M=400, alignment_f='loglikelihood')   
    elif args.selector == 'ACQ_EPIC':
        al = lop.AcquisitionSelection(M=400, alignment_f='epic')
    elif args.selector == 'ACQ_SPEAR':
        al = lop.AcquisitionSelection(M=400, alignment_f='spearman')


    #### create model
    if args.model == 'gp':
        model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7), active_learner=al, normalize_gp=False, use_hyper_optimization=False)
    if args.model == 'linear':
        model = lop.PreferenceLinear(active_learner=al)


    #model.add(np.array([5]), np.array([0.5]), type='abs')
    #

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

