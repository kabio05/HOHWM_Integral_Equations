import pytest
import HOHWM
from numpy import random
import numpy as np
# flake8: noqa


@pytest.mark.parametrize('', [''])
def test_1st_Linear_Fredholm():
    
    f = lambda x: np.exp(x) + np.exp(-x)
    K = lambda x, t: -np.exp(-(x + t))
    u_true = lambda x: np.exp(x)

    test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)
    u_haar_approx = test.solve(approx=True)
    x = HOHWM.collocation()
    err = u_true(x) - u_haar_approx
    assert (err < 1.0e-6).all()


@pytest.mark.parametrize('', [''])
def test_1st_Linear_Volterra():

    f = lambda x: 1/2 * x**2 * np.exp(-x)
    K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
    u_true = lambda x: 1/3 - 1/3 * np.exp(-3/2 * x) * (
        np.cos(np.sqrt(3)/2 * x) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * x))

    x = HOHWM.collocation()
    t = HOHWM.collocation()
    test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)
    u_haar_approx = test.solve(approx=True)
    x = HOHWM.collocation()
    err = u_true(x) - u_haar_approx
    assert (err < 1.0e-6).all()
