import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import sympy as sp
import pandas as pd
import time
import HOHWM
import scipy.optimize as sop


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


# flake8: noqa


def Volterra_1st_iterative_method(
    N, f, K, dK, method="GMRES", tol=1e-8, max_iter=100, verbose=False
):
    def sys_eqs(coefs):
        N = len(coefs)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N)

        coef_haar = coefs.copy()
        C1 = f(0)

        # coef_haar part
        for j in range(N):
            sum_LHS = 0
            sum_LHS = np.sum(
                coef_haar[i] * HOHWM.haar_int_1(x[j], i + 1) for i in range(N)
            )
            sum_u = 0
            sum_K = 0
            for k in range(j + 1):
                sum_u = np.sum(
                    coef_haar[i] * HOHWM.haar_int_1(t[k], i + 1) for i in range(N)
                )
                sum_K += K(x[j], t[k], C1 + sum_u)
            eqs[j] = C1 + sum_LHS - f(x[j]) - 1 / N * sum_K
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

    # get the initial guess
    coefs_init = np.zeros(N)

    # solve the system of equations
    coefs, iter_newton, iter_gmres = newton(
        coefs=coefs_init, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # get the coefficients
    coef_haar = coefs
    C1 = f(0)

    # define approximated function
    def u_haar_approx_func(x):
        # superposition of the Haar wavelet functions
        approx_func_val = C1
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_1(x, k + 1)
        return approx_func_val

    return u_haar_approx_func, iter_newton, iter_gmres



# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________

# 2nd


def Volterra_2nd_iterative_method(
    N, f, K, dK, method="GMRES", tol=1e-8, max_iter=100, verbose=False
):

    def sys_eqs(coefs):
        N = len(coefs) - 2  # Note that coefs includes coef_haar, C1 and C2
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N + 2)

        coefs_haar = coefs[:-2]
        C1 = coefs[-2]
        C2 = coefs[-1]

        # coef_haar part
        sigma_LHS = np.zeros(N)
        sigma_RHS = np.zeros(N)
        for i in range(N):
            sigma_LHS += coefs_haar[i] * HOHWM.haar_int_2(x, i + 1)
        for k in range(N):
            for i in range(N):
                u = C1 + C2 * t[k] + coefs_haar[i] * HOHWM.haar_int_2(t[k], i + 1)
            sigma_RHS += K(x, t[k], u)
        eqs[:-2] = C1 + C2 * x + sigma_LHS - f(x) - 1 / N * sigma_RHS

        # C1 part
        sigma_C1 = 0
        for k in range(N):
            for i in range(N):
                u = C1 + C2 * t[k] + coefs_haar[i] * HOHWM.haar_int_2(t[k], i + 1)
            sigma_C1 += K(0, t[k], u)
        eqs[-2] = C1 - (f(0) + 1 / N * sigma_C1)

        # C2 part
        sigma_LHS = 0
        sigma_RHS = 0
        for i in range(N):
            sigma_LHS += coefs_haar[i] * HOHWM.haar_int_2(1, i + 1)
        for k in range(N):
            for i in range(N):
                u = C1 + C2 * t[k] + coefs_haar[i] * HOHWM.haar_int_2(t[k], i + 1)
            sigma_RHS += K(1, t[k], u)
        eqs[-1] = C1 + C2 + sigma_LHS - f(1) - 1 / N * sigma_RHS

        return eqs

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):

        # iter number
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()

            F = sys_eqs(coefs)
            # J = Jac(coefs)
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

    # get the initial guess
    coefs_init = np.zeros(N + 2)
    coefs_init[-2] = f(0)
    coefs_init[-1] = f(1)

    # solve the system of equations
    coefs, iter_newton, iter_gmres = newton(
        coefs=coefs_init, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # get the coefficients
    coef_haar = coefs[:-2]
    C1 = coefs[-2]
    C2 = coefs[-1]

    # define approximated function
    def u_haar_approx_func(x):
        approx_func_val = C1 + C2 * x
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_2(x, k + 1)
        return approx_func_val

    return u_haar_approx_func, iter_newton, iter_gmres


# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________
# _____________________________________________________________________________

if __name__ == "__main__":

    f = lambda x: np.exp(x) - x * np.exp(-x)
    K = lambda x, t, u: np.exp(-x - 2 * t) * u**2
    u_true = lambda x: np.exp(x)

    K = sp.symbols("K")
    x = sp.symbols("x")
    t = sp.symbols("t")
    u = sp.symbols("u")

    K = sp.exp(-x - 2 * t) * u**2

    # take the derivative of K with respect to u
    dK = sp.diff(K, u)

    # lambdify the derivative of K with respect to u
    K = sp.lambdify((x, t, u), K, "numpy")
    dK = sp.lambdify((x, t, u), dK, "numpy")

    # Compute the error
    print_results = True
    if print_results is True:
        print("Iterative method for Nonlinear Volterra equation")

    col_size = [2, 4, 8, 16, 32, 64]
    err_local = np.zeros(len(col_size))
    err_global = np.zeros(len(col_size))
    iters = np.zeros(len(col_size))
    times = np.zeros(len(col_size))
    methods = ["LU", "GMRES", "LU_sparse"]

    error_data = np.zeros((len(col_size), len(methods)))
    ERC_data = np.zeros((len(col_size) - 1, len(methods)))
    iter_data = np.zeros((len(col_size), len(methods)))
    time_data = np.zeros((len(col_size), len(methods)))

    # open a txt file to store the results
    with open("Newton_Nonlinear_Volterra.txt", "w") as file:
        file.write("Iterative method for Nonlinear Volterra equation\n")
        file.write("\n")

    for s in ["1st"]:
        test_x = 0.5  # calculate the error at x = 0.5
        for method in methods:
            for M in col_size:
                time_start = time.time()
                if s == "1st":
                    u_approx_func, _, iter = Volterra_1st_iterative_method(
                        M,
                        f,
                        K,
                        dK,
                        method=method,
                        tol=1e-8,
                        max_iter=500,
                        verbose=False,
                    )
                elif s == "2nd":
                    u_approx_func, _, iter = Volterra_2nd_iterative_method(
                        M,
                        f,
                        K,
                        dK,
                        method=method,
                        tol=1e-8,
                        max_iter=500,
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
        with open("Newton_Nonlinear_Volterra.txt", "a") as file:
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
