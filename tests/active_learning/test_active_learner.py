# test_active_learner.py
# Written Ian Rankin - December 2023

import pytest

import numpy as np
import lop

def test_active_learner_constructs():
    al = lop.ActiveLearner()
    assert al is not None
    assert isinstance(al, lop.ActiveLearner)

def test_active_learner_prefered_points_func():
    al = lop.ActiveLearner()

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])
    par_ans = [2,3,5]

    prefs = al.get_prefered_set_of_pts(pts)

    assert len(prefs) == pts.shape[0]
    assert prefs == set([0,1,2,3,4,5])


    prefs = al.get_prefered_set_of_pts(pts, 'pareto')

    assert prefs == set(par_ans)


    prefs = al.get_prefered_set_of_pts(pts, 2)

    assert prefs == set([0,1])


    prefs = al.get_prefered_set_of_pts(pts, 0)

    assert prefs == set([0,1,2,3,4,5])

    with pytest.raises(ValueError):
        prefs = al.get_prefered_set_of_pts(pts, -1)

    with pytest.raises(ValueError):
        prefs = al.get_prefered_set_of_pts(pts, 'walks into a bar and order asdfge drinks')



def test_select_crashes_without_input():
    al = lop.ActiveLearner()
    model = lop.Model(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    with pytest.raises(NotImplementedError):
        sel_idxs = al.select(pts, 3)


def test_select():
    al = lop.ActiveLearner()
    model = lop.GP(lop.RBF_kern(1,1), active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    sel_idxs = al.select(pts, 3)

    assert len(sel_idxs) == 3
