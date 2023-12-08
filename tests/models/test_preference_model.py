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
    y_train = np.array([1,2,2,1,-3,-4,0]) 

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
    y_pref = lop.generate_fake_pairs(X_pref, f_sin, 0) + lop.generate_fake_pairs(X_pref, f_sin, 1)

    X_abs = np.array([1.5, 2.5])
    y_abs = f_sin(X_abs)

    pm.add(X_pref, y_pref)

    assert pm is not None

    pm.add(X_abs, y_abs, type='abs')

    assert pm is not None

    # pdb.set_trace()
    # log_like = pm.log_likelyhood_training(np.array([0,0,1,1,1,1]))

    

    # assert not np.isnan(log_like)






