import HOHWM
import numpy as np
import matplotlib.pyplot as plt

# flake8: noqa


def f_ture(x):
    return np.exp(3 * x)


col_size = [1, 2, 4, 8, 16, 32, 64]
ord_text = ["1st derivative", "2nd derivative"]
err_local = np.zeros(len(col_size))
err_global = np.zeros(len(col_size))

for s in [1, 2]:
    for M in col_size:
        # initialize
        N = 2 * M
        x = HOHWM.collocation(N)
        
        if s == 1:
            # solve Ax = b for x to be haar coefficients
            C1 = f_ture(0)
            b_ls = f_ture(x) - C1
            A_ls = HOHWM.haar_int_1_mat(x, N)
            coef_haar = np.linalg.solve(A_ls, b_ls)


            # define approximated function
            def approx_func(x):
                approx_func_val = C1
                for k in range(N):
                    approx_func_val += coef_haar[k] * HOHWM.haar_int_1(x, k + 1)
                return approx_func_val
        
        if s == 2:
            # solve Ax = b for x to be haar coefficients
            b_ls = f_ture(x) - f_ture(0) - (f_ture(1) - f_ture(0)) * x
            A_ls = HOHWM.haar_int_2_mat(x, N)
            p_i2_1 = np.zeros(N)
            for i in range(N):
                p_i2_1[i] = HOHWM.haar_int_2(1, i + 1)
            A_ls = A_ls - np.outer(x, p_i2_1)
            coef_haar = np.linalg.solve(A_ls, b_ls)
            C1 = f_ture(0)
            C2 = f_ture(1) - f_ture(0) - np.dot(coef_haar, p_i2_1)
            
            def approx_func(x):
                approx_func_val = C1 + C2 * x
                for k in range(N):
                    approx_func_val += coef_haar[k] * HOHWM.haar_int_2(x, k + 1)
                return approx_func_val

        # check the convergence speed
        x_test = np.linspace(0, 1, 101)

        loc = 0.5
        f_ture_local = f_ture(loc)
        f_approx_local = approx_func(loc)
        err_local[col_size.index(M)] = np.abs(f_ture_local - f_approx_local)


        f_ture_global = f_ture(x_test)
        f_approx_global = approx_func(x_test)
        err_global[col_size.index(M)] = np.linalg.norm(
            np.abs(f_ture_global - f_approx_global))

    # print the results
    print(ord_text[s - 1], " derivative")
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

    plot = False
    if plot:
        # plot the errors in log-log scale
        plt.figure()
        plt.title(ord_text[s - 1])
        plt.xlabel("log(N)")
        plt.ylabel("log(error)")
        plt.plot(np.log(col_size), np.log(err_local), label="Local error", color="red")
        plt.plot(
            np.log(col_size), np.log(err_global), label="Global error", color="blue"
        )
        plt.legend()
        plt.show()