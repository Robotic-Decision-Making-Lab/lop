# test_training_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities in the training_utility.py function

import pytest
import pdb

import lop
import numpy as np



def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))


def test_k_split_x_y():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


   
    pm.add(X_train, pairs)


    splits = lop.k_fold_x_y(pm.X_train, pm.y_train, 2)
    
    assert np.abs(len(splits[0]) - len(splits[1])) <= 1

def test_k_split_x_y_no_pairs():
    pm = lop.PreferenceModel()


    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sin(X_train)

    pm.add(X_train, y_train, type='abs')

    X_train = np.array([0.7, 6.5])
    y_train = np.array([2, 4])
    pm.add(X_train, y_train, type='ordinal')

    splits = lop.k_fold_x_y(pm.X_train, pm.y_train, 2)

    assert len(splits) == 2
    assert np.abs(len(splits[0]) - len(splits[1])) <= 1

def test_get_y_from_indicies_split():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


   
    pm.add(X_train, pairs)


    splits = lop.k_fold_x_y(pm.X_train, pm.y_train, 2)

    y_0 = lop.get_y_with_idx(pm.y_train, splits[0])

    assert len(y_0[0]) > 0
    assert y_0[1] is None
    assert y_0[2] is None

def test_get_y_from_indicies_multiple_types():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)
    pm.add(X_train, pairs)

    X_train = np.array([0.2,1.5,2.3,3.2,4.2,6.2,7.3])
    y_train = f_sin(X_train)

    pm.add(X_train, y_train, type='abs')

    X_train = np.array([0.7, 6.5])
    y_train = np.array([2, 4])
    pm.add(X_train, y_train, type='ordinal')

    splits = lop.k_fold_x_y(pm.X_train, pm.y_train, 2)
    y_0 = lop.get_y_with_idx(pm.y_train, splits[0])
    y_1 = lop.get_y_with_idx(pm.y_train, splits[1])

    # TODO this test is really weak to be honest.
    assert y_0 is not None
    assert y_1 is not None


