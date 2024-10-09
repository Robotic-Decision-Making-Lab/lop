# test_fake_function.py
# Written Ian Rankin - December 2023
#
# A set of tests of various fake functions used for experiments with lop.

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
    assert z <= 1.0
    assert z == z2

    f.randomize()
    z3 = f(np.array([1.0,0]))

    assert z3 <= 1.0
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

def test_fake_min():
    f = lop.FakeWeightedMin(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z <= 1.0
    assert z == z2


def test_fake_max():
    f = lop.FakeWeightedMax(dimension=2)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert z <= 1.0
    assert z == z2

def test_fake_linear_1d():
    f = lop.FakeLinear(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0

def test_fake_sqared_1d():
    f = lop.FakeSquared(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0

def test_fake_logistic_1d():
    f = lop.FakeLogistic(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0


def test_fake_sin_exp_1d():
    f = lop.FakeSinExp(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0

def test_fake_max_1d():
    f = lop.FakeWeightedMax(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0

def test_fake_min_1d():
    f = lop.FakeWeightedMax(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 <= 1.0




def test_mixture_gaussians():
    f = lop.FakeMixtureGaussian(dimension=1)

    assert f is not None
    assert isinstance(f, lop.FakeFunction)

    f.randomize()

    assert f is not None

    z = f(np.array([1.0,0]))
    z2 = f(np.array([1.0,0]))
    assert len(z) == 2
    assert (z == z2).all

    f.randomize()
    z3 = f(1.0)

    assert z3 >= 0.0