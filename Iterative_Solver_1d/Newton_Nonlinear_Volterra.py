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


def Volterra_1st_iterative_method(
    N, f, K, Phi, method="GMRES", tol=1e-8, max_iter=100, verbose=False
):
    """
    Iterative method for Nonlinear Volterra equation with first order HOHWM, using newton's
    method to solve the system of nonlinear equations.

    Args:
        N (int): number of collocation points
        f (callable function): function f(x) in u(x) = f(x) + int_0^x K(x,t,u(t))dt
        K (callable function): kernel function K(x,t,u)
        method (str, optional): method for solving the linear system. Defaults to "GMRES".
        tol (float, optional): the tolerance for stopping criteria of newton's method. Defaults to 1e-8.
        max_iter (int, optional): the maximum number of iterations for newton's method. Defaults to 100.
        verbose (bool, optional): whether to print the iteration number of newton's
        method and GMRES method. Defaults to False.

    Returns:
        u_haar_approx_func (callable function): the approximated function u_approx(x)
        iter_newton (int): the iteration number of newton's method
        iter_gmres (int): the iteration number of GMRES method
    """

    def sys_eqs(coefs):
        """
        The system of nonlinear equations of the first order HOHWM on nonlinear Volterra equation.
        Check the page 10, equation (56) for the details.

        Args:
            coefs (numpy array): coefficients of the Haar wavelet functions a_1, a_2, ..., a_N

        Returns:
            sys (numpy array): the system of nonlinear equations
        """

        N = len(coefs)
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N)

        coef_haar = coefs.copy()
        C1 = f(0)

        M_A = np.zeros((N, N))
        M_A = HOHWM.haar_int_1_mat(x, N)
        sum_M_A = np.dot(M_A, coef_haar)

        # coef_haar part

        # loop over the x_j, j = 1, 2, ..., N
        for j in range(N):
            sum_LHS = 0
            # this is the sum of the Haar wavelet functions \sum_a_i * \p_i(x_j)
            sum_LHS = sum_M_A[j]

            # initialize the sum_u and sum_K to avoid the error
            sum_u = 0
            sum_K = 0

            for k in range(j): # loop over 0, 1, ..., j-1
                sum_u = sum_M_A[k]
                # we can do this is because x and t are the same in the collocation points
                sum_K += K(x[j], t[k]) * Phi(C1 + sum_u)
            eqs[j] = C1 + sum_LHS - f(x[j]) - 1 / N * sum_K
        return eqs

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):
        """
        Worked newton's method, which has proved to be correct in the previous iterative method

        Args:
            coefs (numpy array): coefficients of the Haar wavelet functions a_1, a_2, ..., a_N
            tol (float, optional): the tolerance for stopping criteria of newton's method. Defaults to tol.
            max_iter (int, optional): the maximum number of iterations for newton's method. Defaults to max_iter.
            method (str, optional): method for solving the linear system. Defaults to method.
            verbose (bool, optional): whether to print the iteration number of newton's method and GMRES method. Defaults to verbose.

        Returns:
            coefs (numpy array): updated coefficients of the Haar wavelet functions a_1, a_2, ..., a_N
            iter_newton (int): the iteration number of newton's method
            iter_gmres (int): the iteration number of GMRES method
        """
        # iter number of newton's method and GMRES method
        iter_newton = 0
        iter_gmres = 0

        for _ in range(max_iter):
            counter = gmres_counter()

            F = sys_eqs(coefs)
            J = sop.approx_fprime(
                coefs, sys_eqs
            )  # we use this to approximate the Jacobian matrix

            if np.linalg.norm(F) < tol or np.linalg.norm(J) < tol:  # stopping criteria
                break

            # use different method to solve the linear system
            if method == "LU":
                delta = np.linalg.solve(J, -F)  # LU
            elif method == "GMRES":
                # here we use restart = len(F) to avoid the bad performance of GMRES
                delta = sla.gmres(J, -F, restart=len(F), callback=counter)[0]
                iter_gmres += counter.niter
            elif method == "LU_sparse":
                delta = sla.spsolve(J, -F)
            else:
                raise ValueError("method can only be LU, GMRES or LU_sparse")

            # update the coefficients
            coefs += delta
            iter_newton += 1

        if verbose:
            print("Newton's method: ", iter_newton, "iterations")
            if method == "GMRES":
                print("GMRES: ", iter_gmres, "iterations")

        if iter_newton == max_iter:
            print("Newton's method does not converge!")

        # for convenience, return the iters_gmres in LU as iter_newton
        # it helps to make the table in below comparison
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
    N, f, K, Phi, method="GMRES", tol=1e-8, max_iter=100, verbose=False
):
    """
    Iterative method for Nonlinear Volterra equation with second order HOHWM, using newton's
    method to solve the system of nonlinear equations.

    Args:
        N (int): number of collocation points
        f (callable function): function f(x) in u(x) = f(x) + int_0^x K(x,t,u(t))dt
        K (callable function): kernel function K(x,t,u)
        method (str, optional): method for solving the linear system. Defaults to "GMRES".
        tol (float, optional): the tolerance for stopping criteria of newton's method. Defaults to 1e-8.
        max_iter (int, optional): the maximum number of iterations for newton's method. Defaults to 100.
        verbose (bool, optional): whether to print the iteration number of newton's
        method and GMRES method. Defaults to False.

    Returns:
        u_haar_approx_func (callable function): the approximated function u_approx(x)
        iter_newton (int): the iteration number of newton's method
        iter_gmres (int): the iteration number of GMRES method
    """

    def sys_eqs(coefs):
        """
        The system of nonlinear equations of the second order HOHWM on nonlinear Volterra equation.
        Check the page 10 and 11, equation (57) and (58) for the details.

        Args:
            coefs (numpy array): coefficients of the Haar wavelet functions a_1, a_2, ..., a_N
            combined with the coefficient C2

        Returns:
            sys (numpy array): the system of nonlinear equations
        """
        N = (
            len(coefs) - 1
        )  # Note that coefs includes coef_haar and C2, hence the length is N+1
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N + 1)

        # get the coefficients
        coefs_haar = coefs[:-1]
        C1 = f(0)
        C2 = coefs[-1]

        M_A = np.zeros((N, N))
        M_A = HOHWM.haar_int_2_mat(x, N)
        sum_M_A = np.dot(M_A, coefs_haar)

        # coef_haar part, this is for the equation (57)
        # similar to the first order, just some changes in additonal constant C2
        for j in range(N):
            sum_LHS = 0
            sum_LHS = sum_M_A[j]
            sum_u = 0
            sum_K = 0
            for k in range(j):
                sum_u = sum_M_A[k]
                sum_K += K(x[j], t[k]) * Phi(C1 + C2 * t[k] + sum_u)
            eqs[j] = C1 + C2 * x[j] + sum_LHS - f(x[j]) - 1 / N * sum_K

        # C2 part, this is for the equation (58)
        # similar to the coef_haar part, just set x = 1
        sum_LHS = 0
        sum_LHS = sum(coefs_haar[i] * HOHWM.haar_int_2(1, i + 1) for i in range(N))
        sum_u = 0
        sum_K = 0
        for k in range(N):
            sum_u = sum(coefs_haar[i] * HOHWM.haar_int_2(t[k], i + 1) for i in range(N))
            sum_K += K(1, t[k]) * Phi(C1 + C2 * t[k] + sum_u)
        eqs[-1] = C1 + C2 * 1 + sum_LHS - f(1) - 1 / N * sum_K

        return eqs

    def newton(coefs, tol=tol, max_iter=max_iter, method=method, verbose=verbose):
        """
        Worked newton's method, which has proved to be correct in the previous iterative method
        Same as the above one, but the system of equations is different

        Args:
            coefs (numpy array): coefficients of the Haar wavelet functions a_1, a_2, ..., a_N
            tol (float, optional): the tolerance for stopping criteria of newton's method. Defaults to tol.
            max_iter (int, optional): the maximum number of iterations for newton's method. Defaults to max_iter.
            method (str, optional): method for solving the linear system. Defaults to method.
            verbose (bool, optional): whether to print the iteration number of newton's method and GMRES method. Defaults to verbose.

        Returns:
            coefs (numpy array): updated coefficients of the Haar wavelet functions a_1, a_2, ..., a_N
            iter_newton (int): the iteration number of newton's method
            iter_gmres (int): the iteration number of GMRES method
        """
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
    coefs_init = np.zeros(N + 1)
    coefs_init[-1] = f(1)

    # solve the system of equations
    coefs, iter_newton, iter_gmres = newton(
        coefs=coefs_init, tol=tol, max_iter=max_iter, method=method, verbose=verbose
    )

    # get the coefficients
    coef_haar = coefs[:-1]
    C1 = f(0)
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

    # the same test function as test 6 in the paper, page 14, equation (68)
    f = lambda x: np.exp(x) - x * np.exp(-x)
    K = lambda x, t: np.exp(-x - 2 * t)
    Phi = lambda u: u**2
    u_true = lambda x: np.exp(x)

    plot = False
    
    if plot is True:
        # plot the true function and approximate function
        col_size = 32
        u_approx_func, _, _ = Volterra_1st_iterative_method(
            col_size, f, K, Phi, method="GMRES", tol=1e-5, max_iter=500, verbose=False
        )
        
        x = np.linspace(0, 1, 1000)
        u_true_all = u_true(x)
        u_haar_approx_all = u_approx_func(x)
        
        plt.plot(x, u_true_all, label="True function")
        plt.plot(x, u_haar_approx_all, label="Approximated function")
        plt.legend()
        plt.show()

    
    # Compute the error
    print_results = True
    if print_results is True:
        print("Iterative method for Nonlinear Volterra equation")

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
    with open("Newton_Nonlinear_Volterra.txt", "w") as file:
        file.write("Iterative method for Nonlinear Volterra equation\n")
        file.write("\n")

    for s in ["1st", "2nd"]:
        test_x = 0.5  # calculate the error at x = 0.5
        for method in methods:
            for M in col_size:
                time_start = time.time()
                if s == "1st":
                    u_approx_func, _, iter = Volterra_1st_iterative_method(
                        M,
                        f,
                        K,
                        Phi,
                        method=method,
                        tol=1e-4,
                        max_iter=500,
                        verbose=False,
                    )
                elif s == "2nd":
                    u_approx_func, _, iter = Volterra_2nd_iterative_method(
                        M,
                        f,
                        K,
                        Phi,
                        method=method,
                        tol=1e-4,
                        max_iter=500,
                        verbose=False,
                    )
                else:
                    raise ValueError("method can only be 1st or 2nd")

                # end time
                time_end = time.time()

                # calculate the local error
                u_true_half = u_true(test_x)
                u_haar_approx_half = u_approx_func(test_x)

                # store the error
                err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)

                # calcualte the global error (not essential)
                # u_true_all = u_true(np.linspace(0, 1, 1000))
                # u_haar_approx_all = u_approx_func(np.linspace(0, 1, 1000))
                # err_global[col_size.index(M)] = np.linalg.norm(
                #     u_true_all - u_haar_approx_all
                # ) / np.sqrt(1000)

                # store the iteration number
                iters[col_size.index(M)] = iter

                # store the time
                times[col_size.index(M)] = time_end - time_start

            # store the error
            error_data[:, methods.index(method)] = err_local
            # error_data[:, methods.index(method)] = err_global

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
