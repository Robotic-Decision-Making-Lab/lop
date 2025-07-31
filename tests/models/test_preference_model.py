# test_user_gp_probits.py
# Written Ian Rankin - October 2022
#
# Test the probit functions for the user GP

import pytest
import lop

import numpy as np


import pdb

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def f_sq(x, data=None):
    return (x/10.0)**2


def test_preference_model_add_pairs():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

   
    pm.add(X_train, pairs)

    assert pm is not None



def test_preference_model_add_ordinal():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    y_train = np.array([1,2,2,1,4,5,3]) 

    pm.add(X_train, y_train, type='ordinal')

    assert pm is not None

    pm.add(X_train, y_train, type='ordinal')
    assert pm is not None

    with pytest.raises(Exception):
        pm.add(X_train, np.array([-1,1,2,4,5,0,1]), type='ordinal')



def test_preference_model_add_abs_bound():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    y_train = np.array([1,2,2,1,3,4,0.1]) / 5.0  

    pm.add(X_train, y_train, type='abs')

    assert pm is not None





def test_preference_model_likelyhood_pairs():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

   
    pm.add(X_train, pairs)
    with pytest.raises(TypeError):
        log_like = pm.log_likelyhood_training()
    log_like = pm.log_likelyhood_training(np.array([0,0,0,1,1,1,0.5]))

    assert not np.isnan(log_like)





def test_adding_multiple_types():
    pm = lop.PreferenceModel(other_probits={'abs': lop.AbsBoundProbit()})

    X_pref = np.array([0,1,2,3])
    y_pref = lop.generate_fake_pairs(X_pref, f_sq, 0) + lop.generate_fake_pairs(X_pref, f_sq, 1)

    X_abs = np.array([1.5, 2.5])
    y_abs = f_sq(X_abs)

    pm.add(X_pref, y_pref)

    assert pm is not None

    pm.add(X_abs, y_abs, type='abs')

    assert pm is not None

def test_preference_model_pairs_second_part_1():
    pm = lop.PreferenceModel()

    pm.add(np.array([0.0]), np.array([0.5]), type='abs')

    X_train = np.array([1.0, 2.0, 3.0])
    #y_train = np.array([0,1,0]) possible values of y_train given preferences
    pairs = [(lop.get_dk(0,1), 0, 1), (lop.get_dk(1,0), 1, 2)]

    pm.add(X_train, pairs)

    # assert the pair indicies are increased with the preference model
    assert pm.y_train[0][0][1] == 1 and pm.y_train[0][0][2] == 2
    assert pm.y_train[0][1][1] == 2 and pm.y_train[0][1][2] == 3

def test_preference_model_pairs_second_part_2():
    pm = lop.PreferenceModel()

    pm.add(np.array([0.0, 5.0, 7.0]), np.array([0.5, 0.7, 0.2]), type='abs')

    X_train = np.array([1.0, 2.0, 3.0])
    #y_train = np.array([0,1,0]) possible values of y_train given preferences
    pairs = [(lop.get_dk(0,1), 0, 1), (lop.get_dk(1,0), 1, 2)]

    pm.add(X_train, pairs)

    # assert the pair indicies are increased with the preference model
    assert pm.y_train[0][0][1] == 3 and pm.y_train[0][0][2] == 4
    assert pm.y_train[0][1][1] == 4 and pm.y_train[0][1][2] == 5

    # test to ensure abs pair indicies are correct
    assert pm.y_train[pm.probit_idxs['abs']][1][0] == 0
    assert pm.y_train[pm.probit_idxs['abs']][1][1] == 1
    assert pm.y_train[pm.probit_idxs['abs']][1][2] == 2

def test_preference_model_pairs_first():
    pm = lop.PreferenceModel()

    

    X_train = np.array([1.0, 2.0, 3.0])
    #y_train = np.array([0,1,0]) possible values of y_train given preferences
    pairs = [(lop.get_dk(0,1), 0, 1), (lop.get_dk(1,0), 1, 2)]

    pm.add(X_train, pairs)
    pm.add(np.array([0.0,4.0]), np.array([0.5, 0.7]), type='abs')

    # assert the pair indicies are increased with the preference model
    assert pm.y_train[pm.probit_idxs['relative_discrete']][0][1] == 0 and pm.y_train[0][0][2] == 1
    assert pm.y_train[0][1][1] == 1 and pm.y_train[0][1][2] == 2

    assert pm.y_train[pm.probit_idxs['abs']][1][0] == 3
    assert pm.y_train[pm.probit_idxs['abs']][1][1] == 4



def test_preference_model_likelyhood_multiple():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sq, 0) + \
            lop.generate_fake_pairs(X_train, f_sq, 1) + \
            lop.generate_fake_pairs(X_train, f_sq, 2) + \
            lop.generate_fake_pairs(X_train, f_sq, 3) + \
            lop.generate_fake_pairs(X_train, f_sq, 4)

   
    pm.add(X_train, pairs)

    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sq(X_train)

    pm.add(X_train, y_train, type='abs')


    with pytest.raises(TypeError):
        log_like = pm.log_likelyhood_training()
    log_like = pm.log_likelyhood_training(np.array([0,0,0,1,1,1,0.5,0.1,0.05,0.2,0.8,0.9,0.9,0.5]))

    assert not np.isnan(log_like)



def test_preference_model_adding_2D_pref():
    pm = lop.PreferenceModel()

    f = lop.FakeFunction(2)

    X_train = np.array([[0,0],[1,2],[2,4],[3,2],[4.2, 5.6],[6,2],[7,8]])
    pairs = lop.generate_fake_pairs(X_train, f, 0) + \
            lop.generate_fake_pairs(X_train, f, 1) + \
            lop.generate_fake_pairs(X_train, f, 2) + \
            lop.generate_fake_pairs(X_train, f, 3) + \
            lop.generate_fake_pairs(X_train, f, 4)

   
    pm.add(X_train, pairs)

    assert pm is not None
    with pytest.raises(TypeError):
        log_like = pm.log_likelyhood_training()
    log_like = pm.log_likelyhood_training(np.array([0,0,0,1,1,1,0.5,0.1,0.05,0.2,0.8,0.9,0.9,0.5]))

    assert not np.isnan(log_like)

def test_preference_model_adding_2D_pref():
    pm = lop.PreferenceModel()

    f = lop.FakeLinear(2)

    X_train = np.array([[0,0],[1,2],[2,4],[3,2],[4.2, 5.6],[6,2],[7,8]])
    pairs = lop.generate_fake_pairs(X_train, f, 0) + \
            lop.generate_fake_pairs(X_train, f, 1) + \
            lop.generate_fake_pairs(X_train, f, 2) + \
            lop.generate_fake_pairs(X_train, f, 3) + \
            lop.generate_fake_pairs(X_train, f, 4)

   
    pm.add(X_train, pairs)

    y_train = f(X_train) / 20.0

    pm.add(X_train, y_train, type='abs')

    with pytest.raises(TypeError):
        log_like = pm.log_likelyhood_training()
    log_like = pm.log_likelyhood_training(np.array([0,0,0,1,1,1,0.5,0.1,0.05,0.2,0.8,0.9,0.9,0.5]))

    assert not np.isnan(log_like)



def test_preference_model_get_set_hyperparameters_pairs():
    m = lop.PreferenceModel()

    p = m.get_hyper()

    assert len(p) == 0

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    m.add(X_train, pairs)

    p = m.get_hyper()

    assert len(p) == 1

def test_preference_model_get_set_hyperparameters_multiple_y_trains():
    m = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)

    m.add(X_train, pairs)
    X_abs = np.array([1.5, 2.5])
    y_abs = f_sq(X_abs)

    m.add(X_abs, y_abs, type='abs')

    p = m.get_hyper()

    assert len(p) == 2

