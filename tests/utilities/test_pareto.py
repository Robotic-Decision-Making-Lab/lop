# test_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities used by the lop algorithms.

import pytest
import pdb

import lop
import numpy as np


def test_pareto_function1():
    rewards = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    pr_idxs = lop.get_pareto(rewards)

    ans = [2,3,5]
    for i in range(len(ans)):
        assert pr_idxs[i] == ans[i]

def test_pareto_function2():
    rewards = np.array([[2,3], [0,0], [4,1]])

    pr_idxs = lop.get_pareto(rewards)

    ans = [0,2]
    for i in range(len(ans)):
        assert pr_idxs[i] == ans[i]