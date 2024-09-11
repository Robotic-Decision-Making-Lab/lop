# test_abs_bayes_info.py
# Written Ian Rankin - July 2024
#
# A set of test function for testing the absloute bayesian model

import pytest

import numpy as np
import lop

import pdb


@pytest.mark.skip(reason="It doens't work")
def test_abs_bayes_info_gain_basic():
    try:
        al = lop.AbsBayesInfo(M=200)
    except:
        print('approxcdf not on this machine, cannot properly test this.')
        return
    model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7), active_learner=al, normalize_gp=False, use_hyper_optimization=False)

    model.add(np.array([5]), np.array([0.5]), type='abs')

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
    x_canidiates = np.array([2.1, 7.5, 0.5, 4.5,5.5,9])

    mu, sigma = model.predict(x_canidiates)
    sel_idx = al.select_greedy(x_canidiates, mu, None, [0,1,2,3,4,5], {})

    assert sel_idx >= 0 and sel_idx < len(x_canidiates)
    assert sel_idx == 5
