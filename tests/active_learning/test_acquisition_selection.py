# test_UCB_learner.py
# Written Ian Rankin - December 2023
#

import pytest

import numpy as np
import lop

import pdb

# the function to approximate
def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_acquisition_selection_constructs():
    al = lop.AcquisitionSelection()
    model = lop.Model(active_learner=al)

    assert isinstance(al, lop.AcquisitionSelection)
    assert isinstance(model, lop.Model)


def test_get_representative_Q():
    al = lop.AcquisitionSelection()
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

    x_train, Q = al.get_representative_Q()

    assert x_train.shape[0] == al.rep_Q_data['num_pts']
    assert Q.shape[0] == al.rep_Q_data['num_Q']
    assert Q.shape[1] == 2



def test_acquisition_selection_basic():
    al = lop.AcquisitionSelection(M=400)
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

    #test_pt_idxs = model.select(x_canidiates, 2)
    
    sel_idx = np.empty(10)
    for i in range(10):
        mu, sigma = model.predict(x_canidiates)
        sel_idx[i] = al.select_greedy(x_canidiates, mu, None, {2,3,4,5}, [0,1])

    uni, counts = np.unique(sel_idx, return_counts=True)
    count_dict = dict(zip(uni, counts))
    assert count_dict[2] > 4

def test_acquisition_selection_loglikelihood():
    al = lop.AcquisitionSelection(M=400, alignment_f='loglikelihood')
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

    sel_idx = np.empty(10)
    for i in range(10):
        mu, sigma = model.predict(x_canidiates)
        sel_idx[i] = al.select_greedy(x_canidiates, mu, None, {2,3,4,5}, [0,1])

    uni, counts = np.unique(sel_idx, return_counts=True)
    count_dict = dict(zip(uni, counts))
    assert count_dict[4] > 2

def test_acquisition_selection_loglikelihood_multiple_to_select_pairs():
    al = lop.AcquisitionSelection(M=400, alignment_f='loglikelihood')
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

    sel_idx = model.select(x_canidiates, 4)
    assert len(sel_idx) == 4


def test_acquisition_selection_empty_model():
    al = lop.AcquisitionSelection(M=400)
    model = lop.PreferenceGP(lop.RBF_kern(0.5,0.7), active_learner=al, normalize_gp=False, use_hyper_optimization=False)

    # carefully selected to have 2.1 and 7.5 (indicies 0 and 1) to be the highest
    # information gain points. (disambiguates which of the two peaks is higher.)
    x_canidiates = np.array([2.1, 7.5, 0.5, 4.5,5.5,9])

    test_pt_idxs = model.select(x_canidiates, 2)

    assert test_pt_idxs is not None





def test_acquisition_selection_bimodal_selection():
    al = lop.AcquisitionSelection(M=400)
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

    sel_idx = np.empty((10, 2))
    for i in range(10):
        sel_idx[i] = model.select(x_canidiates, 2)

    
    cnt = 0
    for i in range(10):
        if 0 in sel_idx[i] and 1 in sel_idx[i]:
            cnt += 1

    assert cnt > 5

def test_acquisition_selection_trains_basic_GP():
    al = lop.AcquisitionSelection()
    model = lop.PreferenceGP(lop.RBF_kern(0.5,1.0), active_learner=al)


    np.random.seed(5) # just to ensure it doesn't break the test on a bad dice roll
    for i in range(10):
        # generate random test set to select test point from
        x_canidiates = np.random.random(20)*10

        test_pt_idxs = model.select(x_canidiates, 2)


        x_train = x_canidiates[test_pt_idxs]
        y_train = f_sin(x_train)
        y_pairs = lop.gen_pairs_from_idx(np.argmax(y_train), list(range(len(y_train))))

        model.add(x_train, y_pairs)


    x_test = np.array([1,2,3,4.5,7,9])
    y_test = f_sin(x_test)
    y_pred = model(x_test)

    assert (np.abs(y_pred - y_test) < 15).all()


@pytest.mark.skip(reason="have not added linear model")
def test_acquisition_selection_trains_linear():
    al = lop.AcquisitionSelection()
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

    assert (np.abs(y_pred - y_test) < 1.0).all()


