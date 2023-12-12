# test_model.py
# Written Ian Rankin - December 2023
#
#

import pytest
import lop

import numpy as np

def test_model_construction():
    m = lop.Model()

    assert m is not None

    with pytest.raises(Exception):
        m(np.array([5,4,3]))

    m = lop.SimplelestModel()

    y = m(np.array([3,4,5]))

    assert y[0] == 3
    assert y[1] == 4
    assert y[2] == 5


