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


def Volterra_2d(N, f, K, Phi, method="LU", tol=1e-8, max_iter=100, verbose=False):

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

        # breakpoint()

        # calculate u_approx_0_y
        u_approx_0_y = np.zeros(N)
        u_approx_0_y = sum_M_E + coefs_const

        # calculate u_approx_x_0
        u_approx_x_0 = np.zeros(N)
        u_approx_x_0 = sum_M_D + coefs_const

        # calculate u_approx_x_y
        u_approx_x_y = np.zeros((N, N))
        # extend sum_M_D to a matrix
        sum_M_D = np.outer(sum_M_D, np.ones(N))
        # extend sum_M_E to a matrix
        sum_M_E = np.outer(np.ones(N), sum_M_E)
        # note coefs_const are added to each element
        u_approx_x_y = sum_M_B + sum_M_D + sum_M_E + coefs_const

        # calculate u_approx_0_0
        u_approx_0_0 = coefs_const

        # first N ** 2 equations, which are u(x, y)
        for m in range(N):
            for n in range(N):
                u_x_m_y_n = 0
                u_x_m_y_n = u_approx_x_y[m, n]

                f_val = f(x[m], y[n])

                K_val = 0
                for p in range(m + 1):
                    for q in range(n + 1):
                        u_s_p_t_q = 0
                        # we can do this is because the collocation points
                        # are the same for x, y and s, t
                        u_s_p_t_q = u_approx_x_y[p, q]
                        K_val += K(x[m], y[n], s[p], t[q]) * Phi(u_s_p_t_q)
                K_val = K_val / N**2

                eqs[m * N + n] = u_x_m_y_n - f_val - K_val

        # N ** 2 to N ** 2 + N eqs, which are u(0, y)
        for n in range(N):
            u_0_y_n = 0
            u_0_y_n = u_approx_0_y[n]

            f_val = f(0, y[n])

            eqs[N**2 + n] = u_0_y_n - f_val

        # N ** 2 + N to N ** 2 + 2N eqs, which are u(x, 0)
        for m in range(N):
            u_x_0_m = 0
            u_x_0_m = u_approx_x_0[m]

            f_val = f(x[m], 0)

            eqs[N**2 + N + m] = u_x_0_m - f_val

        # N ** 2 + 2N + 1 eq, which is u(0, 0)
        u_0_0 = coefs_const
        f_val = f(0, 0)
        eqs[-1] = u_0_0 - f_val

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
            else:
                raise ValueError("method can only be LU, GMRES")

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

        # u_haar_approx = u_haar_approx / N

        return u_haar_approx

    return u_haar_approx_func, iter_newton, iter_gmres


if __name__ == "__main__":

    f = (
        lambda x, y: x * np.sin(y)
        + (x**5) / 4 * (np.cos(y) - 1)
        - ((x**2) / 4) * (np.sin(y) ** 2)
    )
    K = lambda s, t, x, y: x * t**2 + np.cos(s)
    Phi = lambda u: u
    u_true = lambda x, y: x * np.sin(y)

    plot = False

    if plot == True:  # have a visual check

        M = 8

        u_approx_func, _, iter = Volterra_2d(
            M,
            f,
            K,
            Phi,
            method="LU",
            tol=1e-5,
            max_iter=500,
            verbose=False,
        )

        # plot the approximated solution and the true solution
        x = np.linspace(0, 1, 10)
        y = np.linspace(0, 1, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        for i in range(10):
            for j in range(10):
                Z[i, j] = u_approx_func([X[i, j], Y[i, j]])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(X, Y, Z, cmap="viridis")

        u_true_val = np.zeros_like(X)
        for i in range(10):
            for j in range(10):
                u_true_val[i, j] = u_true(X[i, j], Y[i, j])
        ax.plot_surface(X, Y, u_true_val, cmap="viridis")

        plt.show()

    # test the convergence rate

    # Compute the error
    print_results = True
    if print_results is True:
        print("Iterative method for 2D Nonlinear Volterra equation")

    col_size = [2, 4, 8, 16]
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
    with open("2D_Nonlinear_Volterra.txt", "w") as file:
        file.write("Iterative method for 2D Nonlinear Volterra equation\n")
        file.write("\n")

    test_x = [0.5, 0.5]  # test point
    test_region = "global" # "local" or "global"

    for method in methods:

        for M in col_size:
            time_start = time.time()
            u_approx_func, iter, _ = Volterra_2d(
                M,
                f,
                K,
                Phi,
                method=method,
                tol=1e-5,
                max_iter=500,
                verbose=False,
            )
            # end time
            time_end = time.time()
            
            if test_region == "local":
                x = np.linspace(0, 1, 101)
                u_true_half = u_true(test_x[0], test_x[1])
                u_haar_approx_half = u_approx_func(test_x)
                # store the error
                err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)

                iters[col_size.index(M)] = iter
            elif test_region == "global":
                # compute the global error
                x = np.linspace(0, 1, 101)
                y = np.linspace(0, 1, 101)
                X, Y = np.meshgrid(x, y)
                Z = np.zeros_like(X)
                for i in range(101):
                    for j in range(101):
                        Z[i, j] = u_approx_func([X[i, j], Y[i, j]])
                u_true_val = np.zeros_like(X)
                for i in range(101):
                    for j in range(101):
                        u_true_val[i, j] = u_true(X[i, j], Y[i, j])
                err_local[col_size.index(M)] = np.linalg.norm(Z - u_true_val) / np.linalg.norm(u_true_val)
            else:
                raise ValueError("test_region can only be local or global")

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
    with open("2D_Nonlinear_Volterra.txt", "a") as file:
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
