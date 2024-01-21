import HOHWM
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# flake8: noqa

np.set_printoptions(precision=5)

# To test the convergence rate of the iterative method and the direct method

# We first try to implement the iterative method for 1st and 2nd Fredholm integral equation

f = lambda x: np.exp(x) + np.exp(-x)
K = lambda x, t: -np.exp(-(x + t))
u_true = lambda x: np.exp(x)


# 1st


def Fredholm_1st_iterative_method(N, f, K):
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

    # Solve coef_haar iteratively
    # print(len(sp.optimize.fsolve(sys_eqs, np.zeros(N), full_output=True)[1]))
    coef_haar = sp.optimize.fsolve(sys_eqs, np.zeros(N))

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

    return u_haar_approx_func


# ________________________________________________________________________________
# ________________________________________________________________________________


# 2nd
def Fredholm_2nd_iterative_method(N, f, K):
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

    # Solve coef_haar iteratively
    coef_haar = sp.optimize.fsolve(sys_eqs, np.zeros(N))

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

    return u_haar_approx_func


# ________________________________________________________________________________
# ________________________________________________________________________________

N = 16

# plot the approximated function
u_approx_func = Fredholm_2nd_iterative_method(N, f, K)
x = np.linspace(0, 1, 100)
y = u_approx_func(x)
plt.plot(x, y, label="approximation")
plt.plot(x, u_true(x), label="true")
plt.legend()
# plt.show()

# Compute the error
print_results = True
if print_results is True:
    print("Iterative method for linear Fredholm equation")

    col_size = [2, 4, 8, 16, 32, 64]
    err_local = np.zeros(len(col_size))
    err_global = np.zeros(len(col_size))

    for s in ["1st", "2nd"]:
        test_x = 0.5  # calculate the error at x = 0.5

        for M in col_size:
            if s == "1st":
                u_approx_func = Fredholm_1st_iterative_method(M, f, K)
            elif s == "2nd":
                u_approx_func = Fredholm_2nd_iterative_method(M, f, K)
            else:
                raise ValueError("method can only be 1st or 2nd")
            x = np.linspace(0, 1, 101)
            u_true_half = u_true(test_x)
            u_haar_approx_half = u_approx_func(test_x)
            # store the error
            err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)
            # compute the global error with zero-norm
            u_true_vec = u_true(x)
            u_haar_approx_vec = u_approx_func(x)

        # print the error
        print("\n")
        print("Linear Fredholm ({} derivative)".format(s))
        print("Error at {}: ".format(test_x), err_local)
        print(
            "Experimental rate of convergence: ",
            np.diff(np.log(err_local)) / np.log(2),
        )

# Iterative method for linear Fredholm equation


# Linear Fredholm (1st derivative)
# Error at 0.5:  [1.27707e-01 1.42279e-03 3.60181e-04 9.03280e-05 2.25997e-05 5.65103e-06]
# Experimental rate of convergence:  [-6.48797 -1.98193 -1.99548 -1.99887 -1.99972]


# Linear Fredholm (2nd derivative)
# Error at 0.5:  [5.41401e-03 1.42279e-03 3.60181e-04 9.03280e-05 2.25997e-05 5.65103e-06]
# Experimental rate of convergence:  [-1.92797 -1.98193 -1.99548 -1.99887 -1.99972]

# krylov

# Do a table for comparing the LU, conjugate gradient, and GMRES, Minres.
# comparing time and number of interations for interative methods

# mpiexec -n 4
