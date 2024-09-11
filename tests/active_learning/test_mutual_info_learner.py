# test_UCB_learner.py
# Written Ian Rankin - December 2023
#

import pytest

import numpy as np
import lop



# the function to approximate
def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_mutual_info_learner_constructs():
    al = lop.MutualInfoLearner()
    model = lop.Model(active_learner=al)

    assert isinstance(al, lop.MutualInfoLearner)
    assert isinstance(model, lop.Model)

def test_mutual_info_bimodal_selection():
    al = lop.MutualInfoLearner()
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

    test_pt_idxs = model.select(x_canidiates, 2)

    # GV_UCB should always pick the two bimodal points to disambiguate which of the two is larger.
    assert 0 in test_pt_idxs
    assert 1 in test_pt_idxs

def test_mutual_info_learner_trains_basic_GP():
    al = lop.MutualInfoLearner()
    model = lop.GP(lop.RBF_kern(0.5,1.0), active_learner=al)


    np.random.seed(5) # just to ensure it doesn't break the test on a bad dice roll
    for i in range(10):
        # generate random test set to select test point from
        x_canidiates = np.random.random(20)*10

        test_pt_idxs = model.select(x_canidiates, 2)


        x_train = x_canidiates[test_pt_idxs]
        y_train = f_sin(x_train)

        model.add(x_train, y_train)


    x_test = np.array([1,2,3,4.5,7,9])
    y_test = f_sin(x_test)
    y_pred = model(x_test)

    assert (np.abs(y_pred - y_test) < 0.9).all()

def test_mutual_info_trains_linear():
    al = lop.MutualInfoLearner()
    model = lop.PreferenceLinear(active_learner=al)

    f = lop.FakeLinear(2)

    np.random.seed(5) # just to ensure it doesn't break the test on a bad dice roll
    for i in range(10):
        # generate random test set to select test point from
        x_canidiates = np.random.random((20,2))

        test_pt_idxs = model.select(x_canidiates, 2)


        x_train = x_canidiates[test_pt_idxs]
        y_train = f(x_train)
        y_pairs = lop.gen_pairs_from_idx(np.argmax(y_train), list(range(len(y_train))))

        model.add(x_train, y_pairs)


    x_test = np.array([[0,0],[0.8,1.0],[0.6,0.3],[0.7,0.2]])
    y_test = f(x_test)
    y_pred = model(x_test)

    #assert (np.abs(y_pred - y_test) < 1.0).all()
    assert np.argmax(y_pred) == np.argmax(y_test)


