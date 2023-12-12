# test_GP.py
# Written Ian Rankin - December 2023
#
#

import pytest
import lop

import numpy as np

def test_pref_GP_construction():
    gp = lop.PreferenceGP(lop.RBF_kern(1.0, 1.0))

    assert gp is not None


def test_remind_me_to_write_these():
    assert False
