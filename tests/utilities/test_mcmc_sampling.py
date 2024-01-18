# test_utility.py
# Written Ian Rankin - December 2023
#
# A set of tests of various utilities used by the lop algorithms.

import pytest
import pdb

import lop
import numpy as np

def gaussian_liklihood(x):
    return np.log(np.exp(-np.dot(x, x) / 2) / np.sqrt(2 * np.pi))

def test_mcmc_sampling():
    samples = lop.metropolis_hastings(gaussian_liklihood, 500, dim=2)

    assert samples.shape[0] == 500
    assert samples.shape[1] == 2

    assert np.abs(np.mean(samples[:,0])) < 0.5
    assert np.abs(np.mean(samples[:,1])) < 0.5
    assert np.std(samples[:,0]) > 0.5
    assert np.std(samples[:,1]) < 3.0