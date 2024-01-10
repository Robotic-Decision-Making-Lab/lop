# test_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities used by the lop algorithms.

import pytest
import pdb

import lop
import numpy as np


def test_fake_function():
    f = lop.FakeFunction()

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    with pytest.raises(NotImplementedError):
        f(6)


def test_fake_linear():
    f = lop.FakeLinear(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z < 1.0
    assert z == z2

    f.randomize()
    z3 = f(np.array([1.0,0]))

    assert z3 < 1.0
    assert z3 != z


def test_fake_squared():
    f = lop.FakeSquared(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z < 1.0
    assert z == z2

    f.randomize()
    z3 = f(np.array([1.0,0]))

    assert z3 < 1.0
    assert z3 != z

def test_fake_logistic():
    f = lop.FakeLogistic(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z < 1.0
    assert z == z2

    f.randomize()
    z3 = f(np.array([1.0,0]))

    assert z3 < 1.0
    assert z3 != z

def test_fake_sin_exp():
    f = lop.FakeSinExp(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z < 1.0
    assert z == z2

    f.randomize()
    z3 = f(np.array([1.0,0]))

    assert z3 < 1.0
    assert z3 != z