import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import sympy as sp
import pandas as pd
import time
import HOHWM
import scipy.optimize as sop

# flake8: noqa


# For storing the iteration number of GMRES
class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0

    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            # print("iter %3i\trk = %s" % (self.niter, str(rk)))
            pass


def Fredholm_2d(N, f, K, method="LU", tol=1e-8, max_iter=100, verbose=False):

    def sys_eqs(coefs):
        N = (coefs.size - 2) / 2  # coefs will have N+1 + N+1 elements
        N = int(N)
        eqs = np.zeros((N + 1) ** 2)
        x = HOHWM.collocation(N)
        y = HOHWM.collocation(N)
        s = HOHWM.collocation(N)
        t = HOHWM.collocation(N)

        coefs_x = coefs[:N]
        C1 = coefs[N]
        coefs_y = coefs[N + 1 : -1]
        C2 = coefs[-1]

        for m in range(N):
            # coef_haar part
            for j in range(N):
                sum_LHS_u_x = (
                    sum(coefs_x[i] * HOHWM.haar_int_1(x[j], i + 1) for i in range(N))
                    + C1
                )
                sum_LHS_u_y = (
                    sum(coefs_y[i] * HOHWM.haar_int_1(y[j], i + 1) for i in range(N))
                    + C2
                )
                sum_LHS = sum_LHS_u_x * sum_LHS_u_y

                sum_u = 0
                sum_K = 0
                for p in range(N):
                    for q in range(N):
                        sum_u_x = (
                            sum(
                                coefs_x[i] * HOHWM.haar_int_1(x[p], i + 1)
                                for i in range(N)
                            )
                            + C1
                        )
                        sum_u_y = (
                            sum(
                                coefs_y[i] * HOHWM.haar_int_1(y[q], i + 1)
                                for i in range(N)
                            )
                            + C2
                        )
                        sum_u = sum_u_x * sum_u_y
                        sum_K += K(s[m], t[j], x[p], y[q], sum_u)
                eqs[m * (N + 1) + j] = sum_LHS - f(x[m], y[j]) - 1 / (N**2) * sum_K
        return eqs

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):

        # iter number
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()

            F = sys_eqs(coefs)
            J = sop.approx_fprime(coefs, sys_eqs)
            breakpoint()
            if np.linalg.norm(F) < tol or np.linalg.norm(J) < tol:
                break

            if method == "LU":
                delta = np.linalg.solve(J, -F)  # LU
            elif method == "GMRES":
                delta = sla.gmres(J, -F, restart=len(F), callback=counter)[0]
                iter_gmres += counter.niter
            elif method == "LU_sparse":
                delta = sla.spsolve(J, -F)
            else:
                raise ValueError("method can only be LU, GMRES or LU_sparse")

            coefs += delta
            iter_newton += 1

        if verbose:
            print("Newton's method: ", iter_newton, "iterations")
            if method == "GMRES":
                print("GMRES: ", iter_gmres, "iterations")

        if iter_newton == max_iter:
            print("Newton's method does not converge!")

        # for convenience, return the iters_gmres in LU as iter_newton
        if method == "LU":
            iter_gmres = iter_newton
        if method == "LU_sparse":
            iter_gmres = iter_newton
        return coefs, iter_newton, iter_gmres

    # initial guess
    coefs_init = np.zeros(2 * N + 2)

    # solve the system
    coefs, iter_newton, iter_gmres = newton(coefs_init, method=method, verbose=verbose)

    print(coefs, iter_newton, iter_gmres)

    # get coefficients
    coefs_x = coefs[:N]
    C1 = coefs[N]
    coefs_y = coefs[N + 1 : -1]
    C2 = coefs[-1]

    # define approximated function
    def u_haar_approx_func(x):
        u_haar_approx = 0
        for i in range(N):
            for j in range(N):
                u_haar_approx += (coefs_x[i] * HOHWM.haar_int_1(x[0], i + 1) + C1) * (
                    coefs_y[j] * HOHWM.haar_int_1(x[1], j + 1) + C2
                )
        return u_haar_approx

    return u_haar_approx_func, iter_newton, iter_gmres


if __name__ == "__main__":

    f = lambda s, t: np.sin(t) - 1 / 18 * s * t**2 * (
        1 - np.cos(1) * (1 / 2 * np.sin(1) ** 2 + 1)
    )
    K = lambda s, t, x, y, u: 1 / 6 * x * s * t**2 * u**3
    u_true = lambda s, t: np.sin(t)

    M = 2
    
    u_approx_func, _, iter = Fredholm_2d(
        M,
        f,
        K,
        method="LU",
        tol=1e-5,
        max_iter=500,
        verbose=False,
    )
