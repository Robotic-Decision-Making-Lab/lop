# test_active_learner.py
# Written Ian Rankin - December 2023

import pytest

import numpy as np
import lop

import pdb

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

    prefs = al.get_prefered_set_of_pts(pts, {0,2,4})
    assert prefs == set([0,2,4])
    prefs = al.get_prefered_set_of_pts(pts, [0,2,4])
    assert prefs == set([0,2,4])

    with pytest.raises(ValueError):
        prefs = al.get_prefered_set_of_pts(pts, -1)

    with pytest.raises(ValueError):
        prefs = al.get_prefered_set_of_pts(pts, 'walks into a bar and order asdfge drinks')

def test_active_learning_select_best():
    al = lop.ActiveLearner()
    model = lop.Model(active_learner=al)

    pts = np.array([4,6,8,1,0,2,4])
    best_idx = al.select_best(pts,set())
    assert best_idx == 2

    best_idx = al.select_best(pts,{0,4,5})
    assert best_idx == 0

    best_idx = al.select_best(pts,{0,4,5}, {0,4})
    assert best_idx == 5

    best_idx = al.select_best(pts,{0,4,5}, {0,4,5,3})
    assert best_idx == 2

    with pytest.raises(Exception):
        best_idx = al.select_best(pts,{0,4,5}, {0,1,2,3,4,5,6})




def test_select_crashes_without_input():
    al = lop.ActiveLearner()
    model = lop.Model(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    with pytest.raises(NotImplementedError):
        sel_idxs = al.select(pts, 3)

def test_best_learner_constructs():
    al = lop.BestLearner()
    model = lop.Model(active_learner=al)

    assert model is not None
    assert al is not None
    assert model.active_learner is not None


def test_select_basic():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    sel_idxs = al.select(pts, 3)
    assert len(sel_idxs) == 3
    assert sel_idxs[0] == 3

def test_select_pareto():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])
    par_ans = [2,3,5]

    sel_idxs = al.select(pts, 2, prefer_pts='pareto')
    assert len(sel_idxs) == 2
    assert sel_idxs[0] in set(par_ans)
    assert sel_idxs[1] in set(par_ans)


    # test asking for more points then there are pareto optimal points
    sel_idxs = al.select(pts, 4, prefer_pts='pareto')
    assert len(sel_idxs) == 4
    assert sel_idxs[0] in set(par_ans) and sel_idxs[0] == 3
    assert sel_idxs[1] in set(par_ans) and sel_idxs[1] == 5
    assert sel_idxs[2] in set(par_ans) and sel_idxs[2] == 2
    assert sel_idxs[3] not in set(par_ans) and sel_idxs[3] == 0

    # test asking for more points then there are pareto optimal points with previous selected
    sel_idxs = al.select(pts, 3, prefer_pts='pareto', prev_selection=[3,2])
    assert (sel_idxs == np.array([5,0,1])).all()

    # test ensuring pareto optimality is actually doing something
    sel_idxs = al.select(pts, 4)
    assert len(sel_idxs) == 4
    assert sel_idxs[0] in set(par_ans) and sel_idxs[0] == 3
    assert sel_idxs[1] not in set(par_ans) and sel_idxs[1] == 0


def test_select_pareto_with_default():
    al = lop.BestLearner(default_to_pareto=True)
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])
    par_ans = [2,3,5]

    sel_idxs = al.select(pts, 2)
    assert len(sel_idxs) == 2
    assert sel_idxs[0] in set(par_ans)
    assert sel_idxs[1] in set(par_ans)


    # test asking for more points then there are pareto optimal points
    sel_idxs = al.select(pts, 4)
    assert len(sel_idxs) == 4
    assert sel_idxs[0] in set(par_ans) and sel_idxs[0] == 3
    assert sel_idxs[1] in set(par_ans) and sel_idxs[1] == 5
    assert sel_idxs[2] in set(par_ans) and sel_idxs[2] == 2
    assert sel_idxs[3] not in set(par_ans) and sel_idxs[3] == 0


def test_always_select_best():
    al = lop.WorstLearner(always_select_best=True)
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])
    par_ans = [2,3,5]

    sel_idxs = al.select(pts, 2)
    assert (sel_idxs == np.array([3,4])).all()


def test_select_with_arbituary_prefer_pts():
    al = lop.BestLearner(default_to_pareto=True)
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    sel_idxs = al.select(pts, 2, prefer_pts=[1,2])
    assert (sel_idxs == np.array([1,2])).all()

    sel_idxs = al.select(pts, 3, prefer_pts={1,2})
    assert (sel_idxs == np.array([1,2,3])).all()

    # test defaults back to pareto appropriately
    sel_idxs = al.select(pts, 3)
    assert (sel_idxs == np.array([3,5,2])).all()


def test_select_with_previously_selected():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    # test with previously selected
    sel_idxs = al.select(pts, 2, prev_selection=[5,2,1])
    assert (sel_idxs == np.array([3,0])).all()

    # test previously selected + prefered_pts
    sel_idxs = al.select(pts, 2, prev_selection=[5,2,1], prefer_pts=[0])
    assert (sel_idxs == np.array([0,3])).all()

    with pytest.raises(Exception):
        sel_idxs = al.select(pts, 4, prev_selection=[5,2,1], prefer_pts=[0])


def test_select_return_not_selected_basic():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    # test simple
    sel_idxs, not_selected = al.select(pts, 3, return_not_selected=True)
    assert (sel_idxs == np.array([3,0,5])).all()
    assert len(not_selected) == 0

def test_select_return_not_selected_previous_selected():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)
    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    # test with previously selected
    sel_idxs, not_selected = al.select(pts, 2, prev_selection=[5,2,1], return_not_selected=True)
    assert (sel_idxs == np.array([3,0])).all()
    assert len(not_selected) == 0


def test_select_return_not_selected_with_best():
    al = lop.WorstLearner(always_select_best=True)
    model = lop.SimplelestModel(active_learner=al)
    pts = np.array([4,6,8,1,0,2,5])

    sel_idxs, not_selected = al.select(pts, 3, return_not_selected=True)
    assert (sel_idxs == np.array([2,4,3])).all()
    assert (not_selected == np.array([])).all()

    
    sel_idxs, not_selected = al.select(pts, 3, return_not_selected=True, prefer_pts={2,0,5})
    assert (sel_idxs == np.array([2,5,0])).all()
    assert (not_selected == np.array([4,3])).all()

    sel_idxs, not_selected = al.select(pts, 4, return_not_selected=True, prefer_pts={2,0,5})
    assert (sel_idxs == np.array([2,5,0,4])).all()
    assert (not_selected == np.array([3])).all()

    # check if best is not performed if previous points is passed
    sel_idxs, not_selected = al.select(pts, 2, return_not_selected=True, prev_selection={2,0,5})
    assert (sel_idxs == np.array([4,3])).all()
    assert (not_selected == np.array([])).all()

    sel_idxs, not_selected = al.select(pts, 2, return_not_selected=True, prev_selection={3,6}, prefer_pts={0,4})
    assert (sel_idxs == np.array([4,0])).all()
    assert (not_selected == np.array([5])).all()

    sel_idxs, not_selected = al.select(pts, 3, return_not_selected=True, prev_selection={3,6}, prefer_pts={0,4})
    assert (sel_idxs == np.array([4,0,5])).all()
    assert (not_selected == np.array([])).all()


def test_select_return_not_selected_with_pareto_pref():
    al = lop.BestLearner()
    model = lop.SimplelestModel(active_learner=al)

    pts = np.array([[3,4,5], [2, 3,4], [5,2,1], [3,4,6], [3, 2, 1], [2, 7,2]])

    # test basic pareto
    sel_idxs, not_selected = al.select(pts, 3, return_not_selected=True, prefer_pts='pareto')
    assert (sel_idxs == np.array([3,5,2])).all()
    assert (not_selected == np.array([0,1])).all()

    # test pulling not_selected back into solution
    sel_idxs, not_selected = al.select(pts, 4, return_not_selected=True, prefer_pts='pareto')
    assert (sel_idxs == np.array([3,5,2,0])).all()
    assert (not_selected == np.array([1])).all()

    # test pulling not_selected back into solution
    sel_idxs, not_selected = al.select(pts, 5, return_not_selected=True, prefer_pts='pareto')
    assert (sel_idxs == np.array([3,5,2,0,1])).all()
    assert (not_selected == np.array([])).all()
