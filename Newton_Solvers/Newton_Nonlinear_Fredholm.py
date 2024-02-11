import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
import sympy as sp
import pandas as pd
import time
import HOHWM


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


def Fredholm_1st_iterative_method(
    N, f, K, dK, method="GMRES", tol=1e-8, max_iter=100, verbose=False
):
    def sys_eqs(coefs):
        N = len(coefs) - 1  # Note that coefs includes coef_haar and C1
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N + 1)

        coef_haar = coefs[:-1]
        C1 = coefs[-1]

        # coef_haar part
        sigma_LHS = np.zeros(N)
        sigma_RHS = np.zeros(N)
        for i in range(N):
            sigma_LHS += coef_haar[i] * HOHWM.haar_int_1(x, i + 1)
        for k in range(N):
            for i in range(N):
                sigma_RHS += K(
                    x, t[k], C1 + coef_haar[i] * HOHWM.haar_int_1(t[k], i + 1)
                )  # suspicious here
        eqs[:-1] = C1 + sigma_LHS - f(x) - 1 / N * sigma_RHS

        # C1 part
        sigma_C1 = 0
        for k in range(N):
            for i in range(N):
                sigma_C1 += K(
                    0, t[k], C1 + coef_haar[i] * HOHWM.haar_int_1(t[k], i + 1)
                )
        eqs[-1] = C1 - (f(0) + 1 / N * sigma_C1)

        return eqs

    def Jac(coefs):
        N = len(coefs) - 1
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        jac = np.zeros((N + 1, N + 1))

        coef_haar = coefs[:-1]
        C1 = coefs[-1]

        # coef_haar part
        jac[:-1, :-1] = HOHWM.haar_int_1_mat(x, N)
        jac[-1, :-1] = np.zeros(N)  # pi(0) = 0 for all i

        # C1 part
        jac[:, -1] = np.ones(N + 1)

        x = np.append(x, 0)  # for the last row with C1 equation

        RHS_f_ai = np.zeros((N + 1, N))  # J[:, :-1]
        RHS_f_c1 = np.zeros(N + 1)  # J[:, -1]

        for k in range(N):
            dK_chain2 = np.zeros(N)
            u = C1
            for i in range(N):
                haar_value = HOHWM.haar_int_1(t[k], i + 1)
                u += coef_haar[i] * haar_value
                dK_chain2[i] = haar_value  # check this step for specific K
            dK_chain1 = dK(x, t[k], u)
            RHS_f_ai += np.outer(dK_chain1, dK_chain2)  # dK/dai = dK/du * du/dai
            RHS_f_c1 += dK_chain1

        jac[:, :-1] -= RHS_f_ai / N
        jac[:, -1] -= RHS_f_c1 / N

        return jac

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):
        
        # iter number
        iter_newton = 0
        iter_gmres = 0
        
        for _ in range(max_iter):
            counter = gmres_counter()
            
            F = sys_eqs(coefs)
            J = Jac(coefs)
            breakpoint()
            if np.linalg.norm(F) < tol or np.linalg.norm(J) < tol:
                break

            if method == "LU":
                delta = np.linalg.solve(J, -F)  # LU
            elif method == "GMRES":
                delta = sla.gmres(J, -F, restart=len(F), callback=counter)[0]
                iter_gmres += counter.niter
            else:
                raise NotImplementedError("Only support LU, GMRES")

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
    coefs_init = np.zeros(N + 1)
    coefs_init[-1] = f(0)

    # solve the system of equations
    coefs, iter_newton, iter_gmres = newton(
        coefs=coefs_init, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # get the coefficients
    coef_haar = coefs[:-1]
    C1 = coefs[-1]

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


if __name__ == "__main__":

    f = lambda x: np.sin(np.pi * x)
    K = lambda x, t, u: 1 / 5 * np.cos(np.pi * x) * np.sin(np.pi * t) * (u**3)
    u_true = lambda x: np.sin(np.pi * x) + 1 / 3 * (20 - np.sqrt(391)) * np.cos(
        np.pi * x
    )

    K = sp.symbols("K")
    x = sp.symbols("x")
    t = sp.symbols("t")
    u = sp.symbols("u")

    K = 1 / 5 * sp.cos(sp.pi * x) * sp.sin(sp.pi * t) * (u**3)

    # take the derivative of K with respect to u
    dK = sp.diff(K, u)

    # lambdify the derivative of K with respect to u
    K = sp.lambdify((x, t, u), K, "numpy")
    dK = sp.lambdify((x, t, u), dK, "numpy")

    # Compute the error
    print_results = True
    if print_results is True:
        print("Iterative method for Nonlinear Fredholm equation")

    col_size = [2, 4, 8, 16, 32, 64]
    err_local = np.zeros(len(col_size))
    err_global = np.zeros(len(col_size))
    iters = np.zeros(len(col_size))
    times = np.zeros(len(col_size))
    methods = ["LU"]

    error_data = np.zeros((len(col_size), len(methods)))
    ERC_data = np.zeros((len(col_size) - 1, len(methods)))
    iter_data = np.zeros((len(col_size), len(methods)))
    time_data = np.zeros((len(col_size), len(methods)))

    # open a txt file to store the results
    with open("Newton_Nonlinear_Fredholm.txt", "w") as file:
        file.write("Iterative method for Nonlinear Fredholm equation\n")
        file.write("\n")

    for s in ["1st"]:
        test_x = 0.5  # calculate the error at x = 0.5
        for method in methods:
            for M in col_size:
                time_start = time.time()
                if s == "1st":
                    u_approx_func, _, iter = Fredholm_1st_iterative_method(
                        M,
                        f,
                        K,
                        dK,
                        method=method,
                        tol=1e-8,
                        max_iter=200,
                        verbose=False,
                    )
                elif s == "2nd":
                    raise NotImplementedError("2nd is not implemented yet")
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
        with open("Newton_Nonlinear_Fredholm.txt", "a") as file:
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
