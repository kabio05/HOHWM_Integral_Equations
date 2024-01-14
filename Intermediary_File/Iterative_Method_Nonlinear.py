import HOHWM
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# flake8: noqa

np.set_printoptions(precision=5)

f = lambda x: np.sin(np.pi * x)
K = lambda x, t, u: 1 / 5 * np.cos(np.pi * x) * np.sin(np.pi * t) * (u**3)
u_true = lambda x: np.sin(np.pi * x) + 1 / 3 * (20 - np.sqrt(391)) * np.cos(np.pi * x)


# ________________________________________________________________________________
# ________________________________________________________________________________


# 1st


def Fredholm_1st_iterative_method(N, f, K):
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

    # Solve coef_haar and C1 iteratively
    coefs = sp.optimize.fsolve(sys_eqs, np.zeros(N + 1))
    coef_haar = coefs[:-1]
    C1 = coefs[-1]

    # define the approximated function
    def u_haar_approx_func(x):
        # superposition of the Haar wavelet functions
        approx_func_val = C1
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_1(x, k + 1)
        return approx_func_val

    return u_haar_approx_func


# ________________________________________________________________________________
# ________________________________________________________________________________

# 2nd


def Fredholm_2nd_iterative_method(N, f, K):
    def sys_eqs(coefs):
        N = len(coefs) - 2  # Note that coefs includes coef_haar, C1 and C2
        x = HOHWM.collocation(N)
        t = HOHWM.collocation(N)
        eqs = np.zeros(N + 2)

        coef_haar = coefs[:-2]
        C1 = coefs[-2]
        C2 = coefs[-1]

        # coef_haar part
        sigma_LHS = np.zeros(N)
        sigma_RHS = np.zeros(N)
        for i in range(N):
            sigma_LHS += coef_haar[i] * HOHWM.haar_int_2(x, i + 1)
        for k in range(N):
            for i in range(N):
                sigma_RHS += K(
                    x,
                    t[k],
                    C1 + C2 * t[k] + coef_haar[i] * HOHWM.haar_int_2(t[k], i + 1),
                )
        eqs[:-2] = C1 + C2 * x + sigma_LHS - f(x) - 1 / N * sigma_RHS

        # C1 part
        sigma_C1 = 0
        for k in range(N):
            for i in range(N):
                sigma_C1 += K(
                    0,
                    t[k],
                    C1 + C2 * t[k] + coef_haar[i] * HOHWM.haar_int_2(t[k], i + 1),
                )
        eqs[-2] = C1 - (f(0) + 1 / N * sigma_C1)

        # C2 part
        sigma_C2_LHS = 0
        sigma_C2_RHS = 0
        for i in range(N):
            sigma_C2_LHS += coef_haar[i] * HOHWM.haar_int_2(1, i + 1)
        for k in range(N):
            for i in range(N):
                sigma_C2_RHS += K(
                    1,
                    t[k],
                    C1 + C2 * t[k] + coef_haar[i] * HOHWM.haar_int_2(t[k], i + 1),
                )
        eqs[-1] = C1 + C2 + sigma_C2_LHS - f(1) - 1 / N * sigma_C2_RHS

        return eqs

    # Solve coef_haar, C1 and C2 iteratively
    coefs = sp.optimize.fsolve(sys_eqs, np.zeros(N + 2))
    coef_haar = coefs[:-2]
    C1 = coefs[-2]
    C2 = coefs[-1]

    # define the approximated function
    def u_haar_approx_func(x):
        approx_func_val = C1 + C2 * x
        for k in range(N):
            approx_func_val += coef_haar[k] * HOHWM.haar_int_2(x, k + 1)
        return approx_func_val

    return u_haar_approx_func


# ________________________________________________________________________________
# ________________________________________________________________________________

N = 64

# plot the approximated function
u_approx_func = Fredholm_1st_iterative_method(N, f, K)
x = np.linspace(0, 1, 100)
y = u_approx_func(x)
plt.plot(x, y, label="approximation")
plt.plot(x, u_true(x), label="true")
plt.legend()
plt.show()


# Compute the error
print_results = True
if print_results is True:
    print("Iterative method for nonlinear Fredholm equation")

    col_size = [4, 8, 16, 32, 64]
    err_local = np.zeros(len(col_size))
    err_global = np.zeros(len(col_size))

    for s in ["1st"]:
        test_x = 0.25  # calculate the error at x = 0.5

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
        print("Nonlinear Fredholm ({} derivative)".format(s))
        print("Error at {}: ".format(test_x), err_local)
        print(
            "Experimental rate of convergence: ",
            np.diff(np.log(err_local)) / np.log(2),
        )

# Iterative method for nonlinear Fredholm equation


# Nonlinear Fredholm (1st derivative)
# Error at 0.5:  [4.45967e-01 7.70854e-02 1.84180e-02 4.55345e-03 1.13464e-03 2.83107e-04]
# Experimental rate of convergence:  [-2.53241 -2.06535 -2.01608 -2.00472 -2.00282]

# Nonlinear Fredholm (2nd derivative)
# Error at 0.5:  [1.65881e-01 9.51315e-03 8.40976e-03 8.53067e-03 7.26337e-07 4.53688e-08]
# Experimental rate of convergence:  [ -4.12408  -0.17786   0.0206  -13.51973  -4.00087]