# test_gamma_dist.py
# Written Ian Rankin - Febuary 2024
#
# Test the gamma distribution functions

import pytest
import lop

import numpy as np




def test_pdf_gamma_dist():
    x = np.arange(0,10,0.01)

    pdf = lop.pdf_gamma(x, 10, 0.2)

    assert not np.isnan(pdf).any()
    assert x[0] < 0.01


def test_log_pdf_gamma_dist():
    x = np.arange(0,10,0.01)

    pdf = lop.log_pdf_gamma(x, 10, 0.2)

    assert not np.isnan(pdf).any()
    assert pdf[0] < -10


def test_d_log_pdf_gamma_dist():
    x = np.arange(0,10,0.01)

    d_pdf = lop.d_log_pdf_gamma(x, 10, 0.2)

    assert not np.isnan(d_pdf).any()
    assert d_pdf[0] > 0
