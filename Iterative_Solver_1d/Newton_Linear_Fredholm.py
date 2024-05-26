import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import sympy as sp
import pandas as pd
import time
import HOHWM
import pyamg.krylov as pykry

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


def Fredholm_1st_iterative_method(
    N, f, K, method="GMRES", tol=1e-8, max_iter=200, verbose=False
):
    # define the system of equations for the iterative method
    def sys_eqs(coef_haar):
        N = len(coef_haar)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N)

        S_1 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        S_1 = 1 / N * S_1

        S_2 = 0
        for k in range(N):
            S_2 += K(0, t[k])
        S_2 = 1 / N * S_2

        # Haar part
        M_A = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                M_A[:, j] += K(x, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        M_A = HOHWM.haar_int_1_mat(x, N) - 1 / N * M_A

        V_B = np.zeros(N)
        for k in range(N):
            V_B += K(x, t[k])
        V_B = 1 - 1 / N * V_B

        eqs = np.dot(M_A, coef_haar) - (
            f(x) - 1 / (1 - S_2) * (f(0) + np.dot(coef_haar, S_1)) * V_B
        )

        return eqs

    def Jac(coef_haar):
        N = len(coef_haar)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        jac = np.zeros((N, N))

        S_1 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        S_1 = 1 / N * S_1

        S_2 = 0
        for k in range(N):
            S_2 += K(0, t[k])
        S_2 = 1 / N * S_2

        M_A = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                M_A[:, j] += K(x, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        M_A = HOHWM.haar_int_1_mat(x, N) - 1 / N * M_A

        V_B = np.zeros(N)
        for k in range(N):
            V_B += K(x, t[k])
        V_B = 1 - 1 / N * V_B

        jac = M_A + 1 / (1 - S_2) * np.outer(V_B, S_1)

        return jac

    def newton(coef_haar, tol=tol, max_iter=max_iter, method=method, verbose=verbose):
        coef_haar = coef_haar.copy()

        # iter number
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()
            F = sys_eqs(coef_haar)
            J = Jac(coef_haar)
            if np.linalg.norm(F) < tol or np.linalg.norm(J) < tol:
                break

            if method == "LU":
                delta = np.linalg.solve(J, -F)  # LU
            elif method == "GMRES":
                delta = sla.gmres(J, -F, restart=len(F), callback=counter)[0]
                iter_gmres += counter.niter
            elif method == "LU_sparse":
                # turn J into a sparse matrix
                delta = sla.spsolve(J, -F)
            elif method == "GMRES_precon":
                raise NotImplementedError("GMRES_precon is not implemented yet")

            elif method == "BICGSTAB":
                delta = sla.bicgstab(J, -F)[0]
                # breakpoint()

            else:
                raise NotImplementedError("Only support LU, GMRES")

            coef_haar += delta
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

        return coef_haar, iter_newton, iter_gmres

    # initial guess
    coef_haar_initial = np.zeros(N)

    coef_haar, iter_newton, iter_gmres = newton(
        coef_haar_initial, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # compute constant C1
    x = HOHWM.collocation(N)
    t = HOHWM.collocation(N)

    S_1 = np.zeros(N)
    for j in range(N):
        for k in range(N):
            S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
    S_1 = 1 / N * S_1

    S_2 = 0
    for k in range(N):
        S_2 += K(0, t[k])
    S_2 = 1 / N * S_2

    C_1 = 1 / (1 - S_2) * (f(0) + np.dot(coef_haar, S_1))

    # define approximated function
    def u_haar_approx_func(x):
        # superposition of the Haar wavelet functions
        approx_func_val = C_1
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_1(x, k + 1)
        return approx_func_val

    return u_haar_approx_func, iter_newton, iter_gmres


# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________
# 2nd
def Fredholm_2nd_iterative_method(
    N, f, K, method="GMRES", tol=1e-8, max_iter=200, verbose=False
):
    # define the system of equations for the iterative method
    def sys_eqs(coef_haar):
        N = len(coef_haar)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N)

        S_1 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        S_1 = 1 / N * S_1

        S_2 = 0
        for k in range(N):
            S_2 += K(0, t[k])
        S_2 = 1 / N * S_2

        S_3 = 0
        for k in range(N):
            S_3 += K(1, t[k])
        S_3 = 1 / N * S_3

        S_4 = 0
        for k in range(N):
            S_4 += K(0, t[k]) * t[k]
        S_4 = 1 / N * S_4

        S_5 = 0
        for k in range(N):
            S_5 += K(1, t[k]) * t[k]
        S_5 = 1 / N * S_5

        S_6 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_6[j] += K(0, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
        S_6 = 1 / N * S_6

        S_7 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_7[j] += K(1, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
        S_7 = 1 / N * S_7

        S_8 = 1 - S_2 + S_4 * (1 - S_3) - S_5 * (1 - S_2)

        A = f(0) * (1 - S_5) + f(1) * S_4

        D = -f(0) * (1 - S_3) + f(1) * (1 - S_2)

        V_B = np.zeros(N)
        for i in range(N):
            V_B[i] = HOHWM.haar_int_2(1, i + 1)
        V_B -= S_7

        V_E = (1 - S_5) * S_6 - S_4 * V_B

        V_F = (1 - S_3) * S_6 + (1 - S_2) * V_B

        M_A = np.zeros((N, N))
        for i in range(N):
            for k in range(N):
                M_A[:, i] += K(x, t[k]) * HOHWM.haar_int_2(t[k], i + 1)
        M_A = HOHWM.haar_int_2_mat(x, N) - 1 / N * M_A

        V_P = np.zeros(N)
        for k in range(N):
            V_P += K(x, t[k])
        V_P = 1 - 1 / N * V_P

        V_Q = np.zeros(N)
        for k in range(N):
            V_Q += K(x, t[k]) * t[k]
        V_Q = x - 1 / N * V_Q

        C1 = 1 / S_8 * (A + np.dot(coef_haar, V_E))
        C2 = 1 / S_8 * (D - np.dot(coef_haar, V_F))

        eqs = np.dot(M_A, coef_haar) - f(x) + C1 * V_P + C2 * V_Q

        return eqs

    def Jac(coef_haar):
        N = len(coef_haar)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        jac = np.zeros((N, N))

        S_1 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
        S_1 = 1 / N * S_1

        S_2 = 0
        for k in range(N):
            S_2 += K(0, t[k])
        S_2 = 1 / N * S_2

        S_3 = 0
        for k in range(N):
            S_3 += K(1, t[k])
        S_3 = 1 / N * S_3

        S_4 = 0
        for k in range(N):
            S_4 += K(0, t[k]) * t[k]
        S_4 = 1 / N * S_4

        S_5 = 0
        for k in range(N):
            S_5 += K(1, t[k]) * t[k]
        S_5 = 1 / N * S_5

        S_6 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_6[j] += K(0, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
        S_6 = 1 / N * S_6

        S_7 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_7[j] += K(1, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
        S_7 = 1 / N * S_7

        S_8 = 1 - S_2 + S_4 * (1 - S_3) - S_5 * (1 - S_2)

        A = f(0) * (1 - S_5) + f(1) * S_4

        D = -f(0) * (1 - S_3) + f(1) * (1 - S_2)

        V_B = np.zeros(N)
        for i in range(N):
            V_B[i] = HOHWM.haar_int_2(1, i + 1)
        V_B -= S_7

        V_E = (1 - S_5) * S_6 - S_4 * V_B

        V_F = (1 - S_3) * S_6 + (1 - S_2) * V_B

        M_A = np.zeros((N, N))
        for i in range(N):
            for k in range(N):
                M_A[:, i] += K(x, t[k]) * HOHWM.haar_int_2(t[k], i + 1)
        M_A = HOHWM.haar_int_2_mat(x, N) - 1 / N * M_A

        V_P = np.zeros(N)
        for k in range(N):
            V_P += K(x, t[k])
        V_P = 1 - 1 / N * V_P

        V_Q = np.zeros(N)
        for k in range(N):
            V_Q += K(x, t[k]) * t[k]
        V_Q = x - 1 / N * V_Q

        jac = M_A + np.outer(V_P, V_E) / S_8 - np.outer(V_Q, V_F) / S_8

        return jac

    def newton(coef_haar, tol=tol, max_iter=max_iter, method=method, verbose=verbose):
        coef_haar = coef_haar.copy()

        # iter number
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()
            F = sys_eqs(coef_haar)
            J = Jac(coef_haar)

            if np.linalg.norm(F) < tol or np.linalg.norm(J) < tol:
                break

            if method == "LU":
                delta = np.linalg.solve(J, -F)  # LU
            elif method == "GMRES":
                delta = sla.gmres(J, -F, restart=len(F), callback=counter)[0]
                iter_gmres += counter.niter
            elif method == "LU_sparse":
                # turn J into a sparse matrix
                delta = sla.spsolve(J, -F)
            elif method == "GMRES_precon":
                raise NotImplementedError("GMRES_precon is not implemented yet")
                # breakpoint()

            else:
                raise NotImplementedError("Only support LU, GMRES")

            coef_haar += delta
            iter_newton += 1

        if verbose:
            print("Newton's method: ", iter_newton, "iterations")
            if method == "GMRES":
                print("GMRES: ", iter_gmres, "iterations")

        if iter_newton == max_iter:
            print("Newton's method does not converge!")

        # for convenience, return the iters_gmres in LU as 1
        if method == "LU":
            iter_gmres = 1
        if method == "LU_sparse":
            iter_gmres = 1
        return coef_haar, iter_newton, iter_gmres

    coef_haar = np.zeros(N)  # initial guess
    coef_haar, iter_newton, iter_gmeres = newton(
        coef_haar, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # compute constant C1 and C2
    x = HOHWM.collocation(N)
    t = HOHWM.collocation(N)

    S_1 = np.zeros(N)
    for j in range(N):
        for k in range(N):
            S_1[j] += K(0, t[k]) * HOHWM.haar_int_1(t[k], j + 1)
    S_1 = 1 / N * S_1

    S_2 = 0
    for k in range(N):
        S_2 += K(0, t[k])
    S_2 = 1 / N * S_2

    S_3 = 0
    for k in range(N):
        S_3 += K(1, t[k])
    S_3 = 1 / N * S_3

    S_4 = 0
    for k in range(N):
        S_4 += K(0, t[k]) * t[k]
    S_4 = 1 / N * S_4

    S_5 = 0
    for k in range(N):
        S_5 += K(1, t[k]) * t[k]
    S_5 = 1 / N * S_5

    S_6 = np.zeros(N)
    for j in range(N):
        for k in range(N):
            S_6[j] += K(0, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
    S_6 = 1 / N * S_6

    S_7 = np.zeros(N)
    for j in range(N):
        for k in range(N):
            S_7[j] += K(1, t[k]) * HOHWM.haar_int_2(t[k], j + 1)
    S_7 = 1 / N * S_7

    S_8 = 1 - S_2 + S_4 * (1 - S_3) - S_5 * (1 - S_2)

    A = f(0) * (1 - S_5) + f(1) * S_4

    D = -f(0) * (1 - S_3) + f(1) * (1 - S_2)

    V_B = np.zeros(N)
    for i in range(N):
        V_B[i] = HOHWM.haar_int_2(1, i + 1)
    V_B -= S_7

    V_E = (1 - S_5) * S_6 - S_4 * V_B

    V_F = (1 - S_3) * S_6 + (1 - S_2) * V_B

    M_A = np.zeros((N, N))
    for i in range(N):
        for k in range(N):
            M_A[:, i] += K(x, t[k]) * HOHWM.haar_int_2(t[k], i + 1)
    M_A = HOHWM.haar_int_2_mat(x, N) - 1 / N * M_A

    V_P = np.zeros(N)
    for k in range(N):
        V_P += K(x, t[k])
    V_P = 1 - 1 / N * V_P

    V_Q = np.zeros(N)
    for k in range(N):
        V_Q += K(x, t[k]) * t[k]
    V_Q = x - 1 / N * V_Q

    C1 = 1 / S_8 * (A + np.dot(coef_haar, V_E))
    C2 = 1 / S_8 * (D - np.dot(coef_haar, V_F))

    # define approximated function
    def u_haar_approx_func(x):
        approx_func_val = C1 + C2 * x
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_2(x, k + 1)
        return approx_func_val

    return u_haar_approx_func, iter_newton, iter_gmeres


# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________
# ________________________________________________________________________________

if __name__ == "__main__":
    f = lambda x: np.exp(x) + np.exp(-x)
    K = lambda x, t: -np.exp(-(x + t))
    u_true = lambda x: np.exp(x)

    # Compute the error
    print_results = True
    if print_results is True:
        print("Iterative method for linear Fredholm equation")

    col_size = [2, 4, 8, 16, 32, 64]
    err_local = np.zeros(len(col_size))
    err_global = np.zeros(len(col_size))
    iters = np.zeros(len(col_size))
    times = np.zeros(len(col_size))
    methods = ["LU", "GMRES"]

    error_data = np.zeros((len(col_size), len(methods)))
    ERC_data = np.zeros((len(col_size) - 1, len(methods)))
    iter_data = np.zeros((len(col_size), len(methods)))
    time_data = np.zeros((len(col_size), len(methods)))

    # open a txt file to store the results
    with open("Newton_Linear_Fredholm.txt", "w") as file:
        file.write("Iterative method for linear Fredholm equation\n")
        file.write("\n")

    for s in ["1st", "2nd"]:
        test_x = 0.5  # calculate the error at x = 0.5
        for method in methods:
            for M in col_size:
                time_start = time.time()
                if s == "1st":
                    u_approx_func, _, iter = Fredholm_1st_iterative_method(
                        M,
                        f,
                        K,
                        method=method,
                        tol=1e-10,
                        max_iter=100,
                        verbose=False,
                    )
                elif s == "2nd":
                    u_approx_func, _, iter = Fredholm_2nd_iterative_method(
                        M,
                        f,
                        K,
                        method=method,
                        tol=1e-8,
                        max_iter=100,
                        verbose=False,
                    )
                else:
                    raise ValueError("method can only be 1st or 2nd")

                # end time
                time_end = time.time()

                x = np.linspace(0, 1, 101)
                u_true_half = u_true(test_x)
                u_haar_approx_half = u_approx_func(test_x)
                # store the error
                err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)
                # compute the global error with zero-norm
                u_true_vec = u_true(x)
                u_haar_approx_vec = u_approx_func(x)

                iters[col_size.index(M)] = iter

                # store the time
                times[col_size.index(M)] = time_end - time_start

            # store the error
            error_data[:, methods.index(method)] = err_local

            # calculate the experimental rate of convergence
            ERC = np.diff(np.log(err_local)) / np.log(2)
            ERC_data[:, methods.index(method)] = ERC

            # store the iteration number
            iter_data[:, methods.index(method)] = iters

            # store the time
            time_data[:, methods.index(method)] = times

        # convert to pandas dataframe
        df_error = pd.DataFrame(error_data, columns=methods)
        df_error.index = col_size

        df_ERC = pd.DataFrame(ERC_data, columns=methods)
        df_ERC.index = col_size[1:]

        df_iter = pd.DataFrame(iter_data, columns=methods)
        df_iter.index = col_size

        df_time = pd.DataFrame(time_data, columns=methods)
        df_time.index = col_size

        # write the results to txt file
        with open("Newton_Linear_Fredholm.txt", "a") as file:
            file.write("s = {}\n".format(s))
            file.write("\n")
            file.write("\n")

            file.write("Error at x = {}\n".format(test_x))
            file.write(str(df_error))
            file.write("\n")
            file.write("\n")

            file.write("Experimental rate of convergence\n")
            file.write(str(df_ERC))
            file.write("\n")
            file.write("\n")

            file.write("Iteration number\n")
            file.write(str(df_iter))
            file.write("\n")
            file.write("\n")

            file.write("Time\n")
            file.write(str(df_time))
            file.write("\n")
            file.write("\n")

        if print_results is True:
            print("s = ", s)
            print("\n")

            print("Error at x = {}".format(test_x))

            print(df_error)
            print("\n")

            print("Experimental rate of convergence")

            print(df_ERC)
            print("\n")

            print("Iteration number")

            print(df_iter)
            print("\n")

            print("Time")

            print(df_time)
            print("\n")
