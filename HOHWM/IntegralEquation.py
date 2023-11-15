import numpy as np  # noqa
import matplotlib.pyplot as plt # noqa
# flake8: noqa


def index(i):
    """Calculate the step-value of the i-th Haar wavelet."""
    j = int(np.ceil(np.log2(i))) - 1
    k = int(i - 2**j) - 1
    return j, k


def haar_vec(x, i):
    """
    x: input vector
    i: the index of the Haar wavelet function

    return: the Haar wavelet function
    """
    if i == 1:
        return np.ones(len(x))
    if i >= 2:
        j, k = index(i) # j is the scale, k is the translation
        m = 2 ** j
        alpha = k / m
        beta = (k + 0.5) / m
        gamma = (k + 1) / m
        a = 1. * (x>=alpha) * (x<beta) 
        b = -1 * (x>=beta) * (x<=gamma)
        c = a + b
        return c

def haar_int_1(x, i):
    """
    x: input vector
    i: the index of the Haar wavelet function

    return: the integration of the Haar wavelet function
    """
    if i == 1:
        return x
    if i >= 2:
        j, k = index(i) # j is the scale, k is the translation
        m = 2 ** j
        alpha = k / m
        beta = (k + 0.5) / m
        gamma = (k + 1) / m
        a = 1. * (x>=alpha) * (x<beta)
        b = -1. * (x>=beta) * (x<=gamma)
        c = a * (x - alpha) + b * (x - gamma)
        return c


def haar_int_1_mat(x, N):
    mat = np.zeros((N, len(x)))
    for j in range(1, N + 1):
        mat[:, j - 1] = haar_int_1(x, j)
    return mat


def haar_int_2(x, i):
    """
    x: input vector
    i: the index of the Haar wavelet function

    return: the second-order integration of the Haar wavelet function
    """
    if i == 1:
        return 0.5 * x ** 2
    if i >= 2:
        j, k = index(i) # j is the scale, k is the translation
        m = 2 ** j
        alpha = k / m
        beta = (k + 0.5) / m
        gamma = (k + 1) / m
        a = 1. * (x>=alpha) * (x<beta) * (x - alpha) ** 2 / 2
        b = -1. * (x>=beta) * (x<=gamma) * ((x - gamma) ** 2 / 2 - (beta - alpha) ** 2)
        c = 1. * (x>= gamma) * (x <= 1) * (beta - alpha) ** 2
        if i != 0 and (i & (i - 1)) == 0: # if i is power of 2
            c = 0
        int_2 = a + b + c
        return int_2


def haar_int_2_mat(x, N):
    mat = np.zeros((N, len(x)))
    for j in range(1, N + 1):
        mat[:, j - 1] = haar_int_2(x, j)
    return mat


def collocation(N):
    return np.linspace(0, 1, N, endpoint=False) + 0.5 / N


class IntegralEquation:
    """ # noqa
    Initialize an instance of the IntegralEquation class.
    """

    def __init__(self, linear, type, f, K, **kwargs):
        self.linearity = linear
        self.type = type
        self.f = f
        self.K = K

    def solve(self, N, s, plot=False, approx=False, approx_func=False):
        # Make sure N is a power of 2
        N = N
        if N & (N - 1) != 0:
            raise ValueError('N must be a power of 2.')
        
        if self.linearity is True:
            f = self.f
            K = self.K

            t = collocation(N)
            x = collocation(N)

            if self.type == 'Fredholm':
                if s == 1:
                    S_1 = np.zeros(N)
                    for j in range(N):
                        for k in range(N):
                            S_1[j] += K(0, t[k]) * haar_int_1(t[k], j+1)
                    S_1 = 1/N * S_1

                    S_2 = 0
                    for k in range(N):
                        S_2 += K(0, t[k])
                    S_2 = 1/N * S_2

                    M_A = np.zeros((N, N))
                    for j in range(N):
                        for k in range(N):
                            M_A[:, j] += K(x, t[k]) * haar_int_1(t[k], j+1)
                    M_A = haar_int_1_mat(x, N) - 1/N * M_A

                    V_B = np.zeros(N)
                    for k in range(N):
                        V_B += K(x, t[k])
                    V_B = 1 - 1/N * V_B

                    A_ls = M_A + np.outer(V_B, S_1) / (1 - S_2)
                    B_ls = f(x) - f(0) * V_B / (1 - S_2)

                    coef_haar = np.linalg.solve(A_ls, B_ls)

                    # calculate the approximation
                    u_haar_approx = np.zeros(N)
                    for k in range(N):
                        u_haar_approx += coef_haar[k] * haar_int_1(x, k + 1)
                    C_1 = 1 / (1 - S_2) * (f(0) + np.dot(coef_haar, S_1))
                    u_haar_approx += C_1
                    # breakpoint()
                    if plot is True:
                        plt.plot(x, u_haar_approx, label='Approximation')
                    elif approx is True:
                        return u_haar_approx
                    elif approx_func is True:
                        def u_haar_approx_func(x):
                            # superposition of the Haar wavelet functions
                            approx_func_val = C_1
                            for k in range(N):
                                approx_func_val += coef_haar[k] * haar_int_1(x, k + 1)
                            return approx_func_val

                            # interpolation
                            # return np.interp(x, collocation(N), u_haar_approx) 
                        return u_haar_approx_func
                    else:
                        return coef_haar
                elif s == 2:
                    S_1 = np.zeros(N)
                    for j in range(N):
                        for k in range(N):
                            S_1[j] += K(0, t[k]) * haar_int_1(t[k], j+1)
                    S_1 = 1/N * S_1

                    S_2 = 0
                    for k in range(N):
                        S_2 += K(0, t[k])
                    S_2 = 1/N * S_2


                    S_3 = 0
                    for k in range(N):
                        S_3 += K(1, t[k])
                    S_3 = 1/N * S_3

                    S_4 = 0
                    for k in range(N):
                        S_4 += K(0, t[k]) * t[k]
                    S_4 = 1/N * S_4

                    S_5 = 0
                    for k in range(N):
                        S_5 += K(1, t[k]) * t[k]
                    S_5 = 1/N * S_5

                    S_6 = np.zeros(N)
                    for j in range(N):
                        for k in range(N):
                            S_6[j] += K(0, t[k]) * haar_int_2(t[k], j+1)
                    S_6 = 1/N * S_6

                    S_7 = np.zeros(N)
                    for j in range(N):
                        for k in range(N):
                            S_7[j] += K(1, t[k]) * haar_int_2(t[k], j+1)
                    S_7 = 1/N * S_7

                    S_8 = 1 - S_2 + S_4 * (1 - S_3) - S_5 * (1 - S_2)
                    
                    A = f(0) * (1 - S_5) + f(1) * S_4

                    D = -f(0) * (1 - S_3) + f(1) * (1 - S_2)

                    V_B = np.zeros(N)
                    for i in range(N):
                        V_B[i] = haar_int_2(1, i + 1)
                    V_B -= S_7

                    V_E = (1 - S_5) * S_6 - S_4 * V_B

                    V_F = (1 - S_3) * S_6 + (1 - S_2) * V_B
                    
                    M_A = np.zeros((N, N))
                    for j in range(N):
                        for k in range(N):
                            M_A[:, j] += K(x, t[k]) * haar_int_2(t[k], j+1)
                    M_A = haar_int_2_mat(x, N) - 1/N * M_A

                    V_P = np.zeros(N)
                    for k in range(N):
                        V_P += K(x, t[k])
                    V_P = 1 - 1/N * V_P

                    V_Q = np.zeros(N)
                    for k in range(N):
                        V_Q += K(x, t[k]) * t[k]
                    V_Q = x - 1/N * V_Q
                    
                    LHS_ls = M_A + np.outer(V_P, V_E) / S_8 - np.outer(V_Q, V_F) / S_8 # if bug, check here
                    RHS_ls = f(x) - A * V_P / S_8 - D * V_Q / S_8
                    
                    coef_haar = np.linalg.solve(LHS_ls, RHS_ls)
                    
                    u_haar_approx = np.zeros(N)
                    for k in range(N):
                        u_haar_approx += coef_haar[k] * haar_int_2(x, k + 1)
                    C1 = 1 / S_8 * (A + np.dot(coef_haar, V_E))
                    C2 = 1 / S_8 * (D - np.dot(coef_haar, V_F))
                    u_haar_approx += C1 + C2 * x
                    
                    if plot is True:
                        plt.plot(x, u_haar_approx, label='Approximation')
                    elif approx is True:
                        return u_haar_approx
                    elif approx_func is True:
                        def u_haar_approx_func(x):
                            # superposition of the Haar wavelet functions
                            approx_func_val = C1 + C2 * x
                            for k in range(N):
                                approx_func_val += coef_haar[k] * haar_int_2(x, k + 1)
                            return approx_func_val
                        return u_haar_approx_func
                    else:
                        return coef_haar

                else:
                    raise NotImplementedError(
                        "Only s = 1 and s = 2 are implemented."
                    )

            elif self.type == 'Volterra':
                if s == 1:

                    M_A = np.zeros((N, N))
                    for i in range(N):
                        for j in range(N):
                            M_A[i, j] = np.sum(K(x[i], t[:i]) * haar_int_1(t[:i], j+1))
                    M_A = haar_int_1_mat(x, N) - 1/N * M_A

                    V_B = np.zeros(N)
                    for i in range(N):
                        V_B[i] = np.sum(K(x[i], t[:i]))
                    V_B = f(x) - f(0) - f(0) * (1/N * V_B)

                    coef_haar = np.linalg.solve(M_A, V_B)

                    u_haar_approx = np.zeros(N)
                    for k in range(N):
                        u_haar_approx += coef_haar[k] * haar_int_1(x, k + 1)
                    C_1 = f(0)
                    u_haar_approx += C_1

                    if plot is True:
                        plt.plot(x, u_haar_approx, label='Approximation')
                    elif approx is True:
                        return u_haar_approx
                    elif approx_func is True:
                        def u_haar_approx_func(x):
                            # superposition of the Haar wavelet functions
                            # breakpoint()
                            approx_func_val = C_1
                            for k in range(N):
                                approx_func_val += coef_haar[k] * haar_int_1(x, k + 1)
                            return approx_func_val

                            # interpolation
                            # return np.interp(x, collocation(N), u_haar_approx) 
                        return u_haar_approx_func
                    else:
                        return coef_haar
                
                elif s == 2:
                    S_3 = 0
                    for k in range(N):
                        S_3 += K(1, t[k])
                    S_3 = 1/N * S_3

                    S_5 = 0
                    for k in range(N):
                        S_5 += K(1, t[k]) * t[k]
                    S_5 = 1/N * S_5
                    
                    S_7 = np.zeros(N)
                    for j in range(N):
                        for k in range(N):
                            S_7[j] += K(1, t[k]) * haar_int_2(t[k], j+1)
                    S_7 = 1/N * S_7

                    A = -f(0) * (1 - S_3) + f(1)
                    
                    V_B = np.zeros(N)
                    for i in range(N):
                        V_B[i] = haar_int_2(1, i + 1)
                    V_B -= S_7
                    
                    M_A = np.zeros((N, N))
                    for i in range(N):
                        for j in range(N):
                            M_A[i, j] = np.sum(K(x[i], t[:i]) * haar_int_2(t[:i], j+1))
                    M_A = haar_int_2_mat(x, N) - 1/N * M_A

                    V_P = np.zeros(N)
                    for i in range(N):
                        V_P[i] = np.sum(K(x[i], t[:i]))
                    V_P = 1 - 1/N * V_P

                    V_Q = np.zeros(N)
                    for i in range(N):
                        V_Q[i] = np.sum(K(x[i], t[:i]) * t[:i])
                    V_Q = x - 1/N * V_Q

                    LHS_ls = M_A - np.outer(V_Q, V_B) / (1 - S_5)
                    RHS_ls = f(x) - f(0) * V_P - A * V_Q / (1 - S_5)
                    
                    coef_haar = np.linalg.solve(LHS_ls, RHS_ls)
                    # breakpoint()
                    u_haar_approx = np.zeros(N)
                    for k in range(N):
                        u_haar_approx += coef_haar[k] * haar_int_2(x, k + 1)
                    C1 = f(0)
                    C2 = A / (1 - S_5) - np.dot(coef_haar, V_B) / (1 - S_5)
                    u_haar_approx += C1 + C2 * x
                    
                    if plot is True:
                        plt.plot(x, u_haar_approx, label='Approximation')
                    elif approx is True:
                        return u_haar_approx
                    elif approx_func is True:
                        def u_haar_approx_func(x):
                            # superposition of the Haar wavelet functions
                            approx_func_val = C1 + C2 * x
                            for k in range(N):
                                approx_func_val += coef_haar[k] * haar_int_2(x, k + 1)
                            return approx_func_val
                        return u_haar_approx_func
                    else:
                        return coef_haar

                else:
                    raise NotImplementedError(
                        "Only s = 1 and s = 2 are implemented."
                    )

            else:
                raise NotImplementedError(
                    "Only Fredholm and Volterra integral equations are implemented."
                )

        else:
            raise NotImplementedError(
                'Nonlinear integral equations are not implemented yet.'
                )

# N = 64
# f = lambda x: np.exp(x) + np.exp(-x)
# K = lambda x, t: -np.exp(-(x + t))
# test = IntegralEquation(linear=True, type="Fredholm", f=f, K=K)
# test.solve(N=N, s=2, plot=True)
# plt.show()


N = 64
f = lambda x: 1/2 * x**2 * np.exp(-x)
K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
u_true = lambda x: 1/3 - 1/3 * np.exp(-3/2 * x) * (
    np.cos(np.sqrt(3)/2 * x) + np.sqrt(3) * np.sin(np.sqrt(3)/2 * x))

x = collocation(N)
t = collocation(N)
test = IntegralEquation(linear=True, type="Volterra", f=f, K=K)
# u_haar_approx = test.solve(N=N, s=2, approx=True)
# x = collocation(N)
# plt.plot(x, u_haar_approx)
# plt.legend(["Approx"])
# plt.show()
# err = u_true(x) - u_haar_approx
# print(np.linalg.norm(err))

u_approx_func = test.solve(N = N, s=2, approx_func=True)
x = np.linspace(0, 1, 101)
plt.plot(x, u_approx_func(x))
# point x = 0.5 for both functions
plt.plot(0.5, u_approx_func(0.5), 'o', color='black')
plt.show()
u_true_half = u_true(0.5)
u_haar_approx_half = u_approx_func(0.5)
print(abs(u_true_half - u_haar_approx_half))