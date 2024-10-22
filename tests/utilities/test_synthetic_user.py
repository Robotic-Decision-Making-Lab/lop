# test_synthetic_user.py
# Written Ian Rankin - August 2024
#
# A test of the synthetic user class

import pytest
import numpy as np

import lop


def test_perfect_user_choose_2():
    f = lop.FakeLinear(2)
    f.w = np.array([0.5, 0.5])

    us = lop.PerfectUser(f)

    assert us is not None

    r = np.array([[1, 0.5], [0.7, 0.7]])

    sel = us.choose(r)

    assert sel == 0

    sel_pairs = us.choose_pairs(r)

    assert len(sel_pairs) == 1
    assert sel_pairs[0][0] == -1
    assert sel_pairs[0][1] == 0
    assert sel_pairs[0][2] == 1

def test_perfect_user_choose_3():
    f = lop.FakeLinear(2)
    f.w = np.array([0.5, 0.5])

    us = lop.PerfectUser(f)

    assert us is not None

    r = np.array([[1, 0.5], [0.7, 0.7], [0.2, 1.2]])

    sel = us.choose(r)

    assert sel == 0

    sel_pairs = us.choose_pairs(r)

    assert len(sel_pairs) == 2
    assert sel_pairs[0][0] == -1
    assert sel_pairs[0][1] == 0
    assert sel_pairs[0][2] == 1
    assert sel_pairs[1][0] == -1
    assert sel_pairs[1][1] == 0
    assert sel_pairs[1][2] == 2

def test_perfect_user_rate():
    f = lop.FakeLinear(2)
    f.w = np.array([0.5, 0.5])

    us = lop.PerfectUser(f)

    r = np.array([0.7, 1.0])
    rating = us.rate(r)

    assert rating > 0.5 and rating < 0.97

    r = np.array([2000,2000])
    rating = us.rate(r)

    assert rating > 0.9 and rating <= 1.0

    rating = us.rate(-r)
    assert rating >= 0 and rating < 0.1


# def test_human_choice_base():
#     f = lop.FakeLinear(2)
#     f.w = np.array([0.5, 0.5])

#     us = lop.HumanChoiceUser(f)

#     r = np.array([[1, 0.5], [0.7, 0.7]])
#     sel = us.choose(r)

#     assert sel >= 0 and sel <= 1


def test_human_choice_beta_different_p():
    f = lop.FakeWeightedMin(2)
    #f.w = np.array([0.707, 0.707])

    us = lop.HumanChoiceUser(f)

    r_train = np.random.random((100,2))

    N_test = 10000
    r_test = np.random.random((N_test, 2, 2))

    for p_exp in [0.95, 0.9, 0.8, 0.7]:

        us.learn_beta(r_train, p_exp)

        # pairwise comparision
        num_correct = 0
        for i in range(N_test):
            correct = np.argmax(us.fake_f(r_test[i]))
            sel = us.choose(r_test[i])

            if correct == sel:
                num_correct += 1

        p = num_correct / N_test

        assert p > (p_exp-0.04) and p < (p_exp+0.04)


def test_human_choice_beta_opt():
    f = lop.FakeLinear(2)
    f.w = np.array([0.707, 0.707])

    us = lop.HumanChoiceUser(f)

    np.random.seed(0)

    r_train = np.random.random((100,2))
    #r_test = np.random.random((2,2))
    #r_test = np.array([[0.7, 0.9], [0.6, 0.4]])

    N_test = 10000
    r_test = np.random.random((N_test, 2, 2))


    us.learn_beta(r_train, 0.95)

    # pairwise comparision
    num_correct = 0
    for i in range(N_test):
        correct = np.argmax(us.fake_f(r_test[i]))
        sel = us.choose(r_test[i])

        if correct == sel:
            num_correct += 1

    p = num_correct / N_test

    assert p > 0.92 and p < 0.97

    # Rating comparision
    num_correct = 0
    for i in range(N_test):
        correct = np.argmax(us.fake_f(r_test[i]))
        rating = us.rate(r_test[i])

        if correct == np.argmax(rating):
            num_correct += 1

    p = num_correct / N_test

    assert p > 0.92 and p < 0.97


    us.learn_beta(r_train, 0.9)

    num_correct = 0
    for i in range(N_test):
        correct = np.argmax(us.fake_f(r_test[i]))
        sel = us.choose(r_test[i])

        if correct == sel:
            num_correct += 1

    p = num_correct / N_test
    assert p > 0.87 and p < 0.93


def test_human_choice_beta_3_query():
    f = lop.FakeLinear(2)
    f.w = np.array([0.707, 0.707])

    us = lop.HumanChoiceUser(f)

    np.random.seed(0)

    r_train = np.random.random((100,2))
    #r_test = np.random.random((2,2))
    #r_test = np.array([[0.7, 0.9], [0.6, 0.4]])

    N_test = 10000
    r_test = np.random.random((N_test, 3, 2))


    us.learn_beta(r_train, 0.95, Q_size=3)

    num_correct = 0
    for i in range(N_test):
        correct = np.argmax(us.fake_f(r_test[i]))
        sel = us.choose(r_test[i])

        if correct == sel:
            num_correct += 1

    p = num_correct / N_test

    assert p > 0.92 and p < 0.97



    us.learn_beta(r_train, 0.9, Q_size=3)

    num_correct = 0
    for i in range(N_test):
        correct = np.argmax(us.fake_f(r_test[i]))
        sel = us.choose(r_test[i])

        if correct == sel:
            num_correct += 1

    p = num_correct / N_test
    assert p > 0.87 and p < 0.93


def test_human_choice_abs():
    f = lop.FakeLinear(2)
    f.w = np.array([0.5, 0.5])

    us = lop.HumanChoiceUser(f)

    r = np.array([0.7, 1.0])
    rating = us.rate(r)

    assert rating > 0.5 and rating < 0.97

    r = np.array([2000,2000])
    rating = us.rate(r)

    assert rating > 0.9 and rating <= 1.0

    rating = us.rate(-r)
    assert rating >= 0 and rating < 0.1


