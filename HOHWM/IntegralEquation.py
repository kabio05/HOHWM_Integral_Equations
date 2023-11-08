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
        int_1 = haar_int_1(x, i)
        int_2 = np.zeros(len(x))
        for j in range(len(x)):
            int_2[j] = np.trapz(int_1[:j+1], x[:j+1])
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
        self.linear = linear
        self.type = type
        self.f = f
        self.K = K

    def solve(self, N=64, plot=False, approx=False, approx_func=False):
        # Make sure N is a power of 2
        N = N
        if N & (N - 1) != 0:
            raise ValueError('N must be a power of 2.')
        
        if self.linear:
            if plot:
                return self.solve_linear(N=N, plot=True)
            elif approx:
                return self.solve_linear(N=N, approx=True)
            elif approx_func:
                return self.solve_linear(N=N, approx_func=True)
            else:
                return self.solve_linear(N=N)
        else:
            raise NotImplementedError(
                'Nonlinear integral equations are not implemented yet.')

    def solve_linear(self, N, plot=False, approx=False, approx_func=False):
        f = self.f
        K = self.K

        t = collocation(N)
        x = collocation(N)

        if self.type == 'Fredholm':
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

            if plot is True:
                plt.plot(x, u_haar_approx, label='Approximation')
            elif approx is True:
                return u_haar_approx
            elif approx_func is True:
                def u_haar_approx_func(x):
                    return np.interp(x, collocation(N), u_haar_approx)
                return u_haar_approx_func
            else:
                return coef_haar

        elif self.type == 'Volterra':
            M_A = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    for k in range(i):
                        M_A[i, j] += K(x[i], t[k]) * haar_int_1(t[k], j+1)
            M_A = haar_int_1_mat(x, N) - 1/N * M_A

            V_B = np.zeros(N)
            for i in range(N):
                for k in range(i):
                    V_B[i] += K(x[i], t[k])
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
                    return np.interp(x, collocation(N), u_haar_approx)
                return u_haar_approx_func
            else:
                return coef_haar

        else:
            raise NotImplementedError


# f = lambda x: np.exp(x) + np.exp(-x)
# K = lambda x, t: -np.exp(-(x + t))
# test = IntegralEquation(linear=True, type="Fredholm", f=f, K=K)
# test.solve(plot=True)
# plt.show()

# f = lambda x: 1/2 * x**2 * np.exp(-x)
# K = lambda x, t: 1/2 * (x - t)**2 * np.exp(-x + t)
# test = IntegralEquation(linear=True, type="Volterra", f=f, K=K)
# test.solve(approx=True)
# print(test.solve(approx=True))
# plt.plot(collocation(), test.solve(approx=True), label='Approximation')
# plt.show()