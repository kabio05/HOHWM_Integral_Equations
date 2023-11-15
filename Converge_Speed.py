import HOHWM
import numpy as np
import matplotlib.pyplot as plt
# flake8: noqa

# Linear Fredholm (1st derivative)

col_size = [1, 2, 4, 8, 16, 32, 64]
err_local = np.zeros(len(col_size))
err_global = np.zeros(len(col_size))

for test_type in ["Fredholm", "Volterra"]:
    if test_type == "Fredholm":
        f = lambda x: np.exp(x) + np.exp(-x)
        K = lambda x, t: -np.exp(-(x + t))
        u_true = lambda x: np.exp(x)
        test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

    if test_type == "Volterra":
        f = lambda x: 1/2 * x**2 * np.exp(-x)
        K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
        u_true = lambda x: 1/3 - 1/3 * np.exp(-3/2 * x) * (
        np.cos(np.sqrt(3)/2 * x) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * x))
        test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)

    for s in [1, 2]:
        for M in col_size:
            u_approx_func = test.solve(N = 2*M, s=s, approx_func=True)

            x = np.linspace(0, 1, 100)

            # compute the local error at x = 0.5
            u_true_half = u_true(x[50])
            u_haar_approx_half = u_approx_func(x[50])
            # store the error
            err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)

            # compute the global error with zero-norm
            u_true_vec = u_true(x)
            u_haar_approx_vec = u_approx_func(x)
            err_global[col_size.index(M)] = np.linalg.norm(
                np.abs(u_true_vec - u_haar_approx_vec)) # probably wrong

        # print the results
        print("Linear {} ({}st derivative)".format(test_type, s))
        print("Local error: ", err_local)
        print("Global error: ", err_global)
        print("")
        print("Local experimental rate of convergence: ", np.diff(np.log(err_local))/np.log(2))
        print("Global experimental rate of convergence : ", np.diff(np.log(err_global))/np.log(2))
        print("\n")
        # plot the errors in log-log scale
        plt.figure()
        plt.title("Linear {} ({}st derivative)".format(test_type, s))
        plt.xlabel("log(N)")
        plt.ylabel("log(error)")
        plt.plot(np.log(col_size), np.log(err_local), label="Local error", color="red")
        plt.plot(np.log(col_size), np.log(err_global), label="Global error", color="blue")

        # plot two line with slope -1 and -2
        x = np.linspace(0, np.log(col_size)[-1], 100)
        y1 = -2 * x + np.log(err_local[0])
        y2 = -3 * x + np.log(err_global[0])
        plt.plot(x, y1, label="slope -2", linestyle="dashed", color="grey")
        plt.plot(x, y2, label="slope -3", linestyle="dashed", color="black")

        plt.legend()
        plt.savefig("Linear_{}_{}st_derivative.png".format(test_type, s), dpi=300)




