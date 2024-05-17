# characterize_cdf.py
# Written Ian Rankin - May 2024
#
# a set of functions to evaluate the approximate cdf function
# Trying to determine how the approximate cdf functions perform.

import numpy as np
import matplotlib.pyplot as plt
import lop
import time




def calc_prob_B_test(methods):
    al = lop.BayesInfoGain()

    model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7), active_learner=al, normalize_gp=False, use_hyper_optimization=False)

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

    # carefully selected to have 2.1 and 7.5 (indicies 0 and 1) to be the highest
    # information gain points. (disambiguates which of the two peaks is higher.)
    x_canidiates = np.array([2.1, 7.5, 0.5, 4.5,5.5,9.2])

    mu = model(x_canidiates)

    for i, method in enumerate(methods):

        t_start = time.time()
        for j in range(50):
            p = al.p_B_pref_gp(x_canidiates, mu, cdf_method=method)
        t_end = time.time()
        print(method + ' time taken: ' + str(t_end - t_start))

        plt.bar(np.arange(len(p))+i*0.2, p, width=0.2)

    plt.legend(methods)

    plt.show()





def main():
    cov = np.array([[1.0, 0.2,-0.1],
                    [0.2, 0.5, 0.3],
                    [-0.1, 0.3, 1.3]])
    mu = np.array([0, -0.5, 0.2])


    methods = ['full', 'switch', 'independent', 'mvn']

    for method in methods:
        print(method)
        p = lop.calc_cdf(mu, cov)

        
        print(p)


    calc_prob_B_test(methods)







    


if __name__ == '__main__':
    main()