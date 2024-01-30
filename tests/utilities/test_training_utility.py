# test_training_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities in the training_utility.py function

import pytest
import pdb

import lop
import numpy as np


def test_k_split_half_simple():
    test_data = [[1,2,3,4,5,6,7,8],[],[]]

    splits = lop.k_fold_split(test_data, 2)

    assert len(splits) == 2
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3

    assert len(splits[0][0]) == 4
    assert len(splits[1][0]) == 4

def test_k_split_half_multiple_sets():
    test_data = [[1,2,3,4,5,6,7,8],[9,10],[11,12,13]]

    splits = lop.k_fold_split(test_data, 2)

    assert len(splits) == 2
    assert len(splits[0]) == 3
    assert len(splits[1]) == 3

    assert len(splits[0][0]) == 4
    assert len(splits[1][0]) == 4

    assert len(splits[0][1]) == 1
    assert len(splits[1][1]) == 1

    assert len(splits[0][2]) == 2
    assert len(splits[1][2]) == 1

def f_sin(x, data=None):
    return 2 * np.cos(np.pi * (x-2)) * np.exp(-(0.9*x))

def test_k_split_preference_model_just_pairs():
    pm = lop.PreferenceModel()

    X_train = np.array([0,1,2,3,4.2,6,7])
    pairs = lop.generate_fake_pairs(X_train, f_sin, 0) + \
            lop.generate_fake_pairs(X_train, f_sin, 1) + \
            lop.generate_fake_pairs(X_train, f_sin, 2) + \
            lop.generate_fake_pairs(X_train, f_sin, 3) + \
            lop.generate_fake_pairs(X_train, f_sin, 4)


   
    pm.add(X_train, pairs)


    splits = lop.k_fold_split(pm.y_train,3)

    assert len(splits) == 3

    for i in range(2):
        for j in range(i+1,3):
            assert np.abs(len(splits[i][0]) - len(splits[j][0])) <= 1

def test_k_split_preference_model_multiple_types():
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

    splits = lop.k_fold_split(pm.y_train,3)

    assert len(splits) == 3

    for i in range(2):
        for j in range(i+1,3):
            for k in [0,2]:
                assert np.abs(len(splits[i][k]) - len(splits[j][k])) <= 1




