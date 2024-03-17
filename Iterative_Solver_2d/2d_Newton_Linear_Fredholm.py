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


def Fredholm_2d(N, f, K, Phi, method="LU", tol=1e-8, max_iter=100, verbose=False):

    def sys_eqs(coefs):
        N = np.sqrt(len(coefs)) - 1  # note coefs is N ** 2 + 2N + 1 array
        N = int(N)
        eqs = np.zeros((N + 1) ** 2)
        x = HOHWM.collocation(N)
        y = HOHWM.collocation(N)
        s = HOHWM.collocation(N)
        t = HOHWM.collocation(N)

        coefs_b = coefs[: N**2]
        coefs_d = coefs[N**2 : N**2 + N]
        coefs_e = coefs[N**2 + N : N**2 + 2 * N]
        coefs_const = coefs[-1]

        # store coefs_b in matrix form
        coefs_M_b = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                coefs_M_b[i, j] = coefs_b[i * N + j]

        # first N ** 2 equations, which are u(x, y)

        # to avoid repeated calculation, we introduce
        # some temporary variables
        sum_M_B = np.zeros((N, N))
        sum_M_D = np.zeros(N)
        sum_M_E = np.zeros(N)

        # E_1 term in u_approx
        for i in range(N):
            for j in range(N):
                p_i_x = HOHWM.haar_int_1(x, i + 1)  # vector
                p_j_y = HOHWM.haar_int_1(y, j + 1)  # vector
                ###### maybe wrong, CHECK #####
                sum_M_B += coefs_M_b[i, j] * np.outer(p_i_x, p_j_y)  # matrix

        # E_2 term in u_approx
        M_D = np.zeros((N, N))
        M_D = HOHWM.haar_int_1_mat(x, N)  # matrix
        sum_M_D = np.dot(M_D, coefs_d)  # N * 1 vector

        # E_3 term in u_approx
        M_E = np.zeros((N, N))
        M_E = HOHWM.haar_int_1_mat(y, N)  # matrix
        sum_M_E = np.dot(M_E, coefs_e)  # 1 * N vector

        # calculate u_approx
        u_approx_x_y = np.zeros((N, N))
        # extend sum_M_D to a matrix
        sum_M_D = np.outer(sum_M_D, np.ones(N))
        # extend sum_M_E to a matrix
        sum_M_E = np.outer(np.ones(N), sum_M_E)
        # note coefs_const are added to each element
        u_approx_x_y = sum_M_B + sum_M_D + sum_M_E + coefs_const

        for m in range(N):
            for n in range(N):
                u_x_m_y_n = 0
                for i in range(N):
                    for j in range(N):
                        u_x_m_y_n += (
                            coefs_b[i * N + j]
                            * HOHWM.haar_int_1(x[m], i + 1)
                            * HOHWM.haar_int_1(y[n], j + 1)
                        )
                        u_x_m_y_n += coefs_e[j] * HOHWM.haar_int_1(y[n], j + 1)
                    u_x_m_y_n += coefs_d[i] * HOHWM.haar_int_1(x[m], i + 1)
                u_x_m_y_n += coefs_const

                f_val = f(x[m], y[n])

                K_val = 0
                for p in range(N):
                    for q in range(N):
                        u_s_p_t_q = 0
                        for i in range(N):
                            for j in range(N):
                                u_s_p_t_q += (
                                    coefs_b[i * N + j]
                                    * HOHWM.haar_int_1(s[p], i + 1)
                                    * HOHWM.haar_int_1(t[q], j + 1)
                                )
                                u_s_p_t_q += coefs_e[j] * HOHWM.haar_int_1(t[q], j + 1)
                            u_s_p_t_q += coefs_d[i] * HOHWM.haar_int_1(s[p], i + 1)
                        u_s_p_t_q += coefs_const
                        K_val += K(x[m], y[n], s[p], t[q]) * Phi(u_s_p_t_q)
                K_val = K_val / N**2

                eqs[m * N + n] = u_x_m_y_n - f_val - K_val

        # N ** 2 to N ** 2 + N eqs, which are u(0, y)
        for n in range(N):
            u_0_y_n = 0
            for j in range(N):
                u_0_y_n += coefs_e[j] * HOHWM.haar_int_1(y[n], j + 1)
            u_0_y_n += coefs_const

            f_val = f(0, y[n])

            K_val = 0
            for p in range(N):
                for q in range(N):
                    u_s_p_t_q = 0
                    for i in range(N):
                        for j in range(N):
                            u_s_p_t_q += (
                                coefs_b[i * N + j]
                                * HOHWM.haar_int_1(s[p], i + 1)
                                * HOHWM.haar_int_1(t[q], j + 1)
                            )
                            u_s_p_t_q += coefs_e[j] * HOHWM.haar_int_1(t[q], j + 1)
                        u_s_p_t_q += coefs_d[i] * HOHWM.haar_int_1(s[p], i + 1)
                    u_s_p_t_q += coefs_const
                    K_val += K(0, y[n], s[p], t[q]) * Phi(u_s_p_t_q)
            K_val = K_val / N**2
            eqs[N**2 + n] = u_0_y_n - f_val - K_val

        # N ** 2 + N to N ** 2 + 2N eqs, which are u(x, 0)
        for m in range(N):
            u_x_0_n = 0
            for i in range(N):
                u_x_0_n += coefs_d[i] * HOHWM.haar_int_1(x[m], i + 1)
            u_x_0_n += coefs_const

            f_val = f(x[m], 0)

            K_val = 0
            for p in range(N):
                for q in range(N):
                    u_s_p_t_q = 0
                    for i in range(N):
                        for j in range(N):
                            u_s_p_t_q += (
                                coefs_b[i * N + j]
                                * HOHWM.haar_int_1(s[p], i + 1)
                                * HOHWM.haar_int_1(t[q], j + 1)
                            )
                            u_s_p_t_q += coefs_e[j] * HOHWM.haar_int_1(t[q], j + 1)
                        u_s_p_t_q += coefs_d[i] * HOHWM.haar_int_1(s[p], i + 1)
                    u_s_p_t_q += coefs_const
                    K_val += K(x[m], 0, s[p], t[q], u_s_p_t_q)
            K_val = K_val / N**2
            eqs[N**2 + N + m] = u_x_0_n - f_val - K_val
            # print(eqs[N**2 + N: N**2 + 2*N])

        # N ** 2 + 2N + 1 eq, which is u(0, 0)
        u_0_0 = coefs_const
        f_val = f(0, 0)
        K_val = 0
        for p in range(N):
            for q in range(N):
                u_s_p_t_q = 0
                for i in range(N):
                    for j in range(N):
                        u_s_p_t_q += (
                            coefs_b[i * N + j]
                            * HOHWM.haar_int_1(s[p], i + 1)
                            * HOHWM.haar_int_1(t[q], j + 1)
                        )
                        u_s_p_t_q += coefs_e[j] * HOHWM.haar_int_1(t[q], j + 1)
                    u_s_p_t_q += coefs_d[i] * HOHWM.haar_int_1(s[p], i + 1)
                u_s_p_t_q += coefs_const
                K_val += K(0, 0, s[p], t[q], u_s_p_t_q)
        K_val = K_val / N**2
        eqs[-1] = u_0_0 - f_val - K_val

        return eqs

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):

        # iter number
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()

            F = sys_eqs(coefs)
            J = sop.approx_fprime(coefs, sys_eqs)
            # breakpoint()
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
    coefs_init = np.zeros(N**2 + 2 * N + 1)

    # solve the system
    coefs, iter_newton, iter_gmres = newton(coefs_init, method=method, verbose=verbose)

    # get coefficients
    coefs_b = coefs[: N**2]
    coefs_d = coefs[N**2 : N**2 + N]
    coefs_e = coefs[N**2 + N : N**2 + 2 * N]
    coefs_const = coefs[-1]

    # define approximated function
    def u_haar_approx_func(x):
        u_haar_approx = 0
        for i in range(N):
            for j in range(N):
                u_haar_approx += (
                    coefs_b[i * N + j]
                    * HOHWM.haar_int_1(x[0], i + 1)
                    * HOHWM.haar_int_1(x[1], j + 1)
                )
                u_haar_approx += coefs_e[j] * HOHWM.haar_int_1(x[1], j + 1)
            u_haar_approx += coefs_d[i] * HOHWM.haar_int_1(x[0], i + 1)
        u_haar_approx += coefs_const
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
