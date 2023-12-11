import HOHWM
import numpy as np
import matplotlib.pyplot as plt

# flake8: noqa

# Linear Fredholm (1st derivative)

col_size = [1, 2, 4, 8, 16, 32, 64]
err_local = np.zeros(len(col_size))
err_global = np.zeros(len(col_size))
s_text = ["1st", "2nd"]

for test_type in ["Fredholm"]:
    if test_type == "Fredholm":
        # f = lambda x: np.exp(x) + np.exp(-x)
        # K = lambda x, t: -np.exp(-(x + t))
        # u_true = lambda x: np.exp(x)
        # test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

        # f = lambda x: 5/6 * x - 1/9
        # K = lambda x, t: 1/3 * (x + t)
        # u_true = lambda x: x
        # test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

        # f = lambda x: 2/3 * x
        # K = lambda x, t: x * t
        # u_true = lambda x: x
        # test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

        # f = lambda x: 9/10 * x**2
        # K = lambda x, t: 1/2 * (x**2 * t**2)
        # u_true = lambda x: x**2
        # test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

        # f = lambda x: np.exp(x) - 1
        # K = lambda x, t: t
        # u_true = lambda x: np.exp(x)
        # test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

        f = lambda x: np.exp(3 * x) - 1 / 9 * (2 * np.exp(3) + 1) * x
        K = lambda x, t: x * t
        u_true = lambda x: np.exp(3 * x)
        test = HOHWM.IntegralEquation(linear=True, type="Fredholm", f=f, K=K)

    if test_type == "Volterra":
        f = lambda x: 1/2 * x**2 * np.exp(-x)
        K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
        u_true = lambda x: 1/3 - 1/3 * np.exp(-3/2 * x) * (
        np.cos(np.sqrt(3)/2 * x) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * x))
        test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)

        # f = lambda x: x
        # K = lambda x, t: -(x - t)
        # u_true = lambda x: np.sin(x)
        # test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)

        # f = lambda x: 2 * np.cosh(x) - x * np.sinh(x) - 1
        # K = lambda x, t: t
        # u_true = lambda x: np.cosh
        # test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)

        # f = lambda x: x + 1
        # K = lambda x, t: x - t
        # u_true = lambda x: np.exp(x)
        # test = HOHWM.IntegralEquation(linear=True, type="Volterra", f=f, K=K)

    for s in [1, 2]:
        for M in col_size:
            u_approx_func = test.solve(N=2 * M, s=s, approx_func=True, method="ls")
            x = np.linspace(0, 1, 101)
            # print("x", x)
            # x = x[1:-1]
            # print("x", x)
            # compute the local error at x = 0.5
            u_true_half = u_true(0.5)
            u_haar_approx_half = u_approx_func(0.5)
            # store the error
            err_local[col_size.index(M)] = abs(u_true_half - u_haar_approx_half)
            # compute the global error with zero-norm
            u_true_vec = u_true(x)
            u_haar_approx_vec = u_approx_func(x)
            # err_global[col_size.index(M)] = np.sqrt(np.sum(np.power(np.abs(u_true_vec - u_haar_approx_vec), 2)))
            err_global[col_size.index(M)] = np.linalg.norm(
                np.abs(u_true_vec - u_haar_approx_vec)
            )  # probably wrong
        # err_global /= (2 * M * 100)

        # print the results
        print("Linear {} ({}st derivative)".format(test_type, s))
        print("Local error: ", err_local)
        print("Global error: ", err_global)
        print("")
        print(
            "Local experimental rate of convergence: ",
            np.diff(np.log(err_local)) / np.log(2),
        )
        print(
            "Global experimental rate of convergence : ",
            np.diff(np.log(err_global)) / np.log(2),
        )
        print("\n")
        # plot the errors in log-log scale
        plt.figure()
        plt.title("Linear {} ({} derivative)".format(test_type, s_text[s - 1]))
        plt.xlabel("log(N)")
        plt.ylabel("log(error)")
        plt.plot(np.log(col_size), np.log(err_local), label="Local error", color="red")
        plt.plot(
            np.log(col_size), np.log(err_global), label="Global error", color="blue"
        )

        if s == 1:
            x = np.linspace(0, np.log(col_size)[-1], 100)
            y1 = -2 * x + np.log(err_local[0])
            y2 = -2 * x + np.log(err_global[0])
            plt.plot(x, y1, label="slope -2", linestyle="dashed", color="grey")
            plt.plot(x, y2, label="slope -2", linestyle="dashed", color="black")
        if s == 2:
            x = np.linspace(0, np.log(col_size)[-1], 100)
            y1 = -4 * x + np.log(err_local[0])
            y2 = -3 * x + np.log(err_global[0])
            plt.plot(x, y1, label="slope -4", linestyle="dashed", color="grey")
            plt.plot(x, y2, label="slope -3", linestyle="dashed", color="black")

        plt.legend()
        plt.savefig(
            "Linear_{}_{}_derivative.png".format(test_type, s_text[s - 1]), dpi=300
        )
