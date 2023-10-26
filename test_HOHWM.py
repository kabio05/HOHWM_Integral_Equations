import pytest
import HOHWM
from numpy import random
import numpy as np
# flake8: noqa


@pytest.mark.parametrize('N', [4, 8, 16, 32, 64, 128])
def test_1st_Linear_Fredholm(N):
    N = N
    f = lambda x: np.exp(x) + np.exp(-x)
    K = lambda x, t: -np.exp(-(x + t))
    u_true = lambda x: np.exp(x)

    test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)
    u_haar_approx = test.solve(N=N, approx=True)
    x = HOHWM.collocation(N)
    err = u_true(x) - u_haar_approx
    assert (err < 1.0e-8).all()


@pytest.mark.parametrize('N', [4, 8, 16, 32, 64, 128])
def test_1st_Linear_Volterra(N):
    N = N
    f = lambda x: 1/2 * x**2 * np.exp(-x)
    K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
    u_true = lambda x: 1/3 - 1/3 * np.exp(-3/2 * x) * (
        np.cos(np.sqrt(3)/2 * x) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * x))

    x = HOHWM.collocation(N)
    t = HOHWM.collocation(N)
    test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)
    u_haar_approx = test.solve(N=N, approx=True)
    x = HOHWM.collocation(N)
    err = u_true(x) - u_haar_approx
    assert (err < 1.0e-8).all()
