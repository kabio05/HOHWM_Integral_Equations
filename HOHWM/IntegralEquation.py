import numpy as np  # noqa


def index(i):
    """
    Calculate the step-value of the i-th Haar wavelet
    """
    j = int(np.ceil(np.log2(i))) - 1
    k = int(i - 2**j) - 1
    return j, k


def haar_int_1(x, i):
    if i == 1:
        return x
    if i >= 2:
        j, k = index(i)  # j is the scale, k is the translation
        m = 2**j
        alpha = k / m
        beta = (k + 0.5) / m
        gamma = (k + 1) / m
        a = (x >= alpha) & (x < beta)
        b = (x >= beta) & (x <= gamma)
        b = b.astype(int)
        a = a.astype(int)
        c = a * (x - alpha) - b * (x - gamma)
        return c


def haar_int_1_mat(x, N):
    mat = np.zeros((N, len(x)))
    for j in range(1, N + 1):
        mat[:, j - 1] = haar_int_1(x, j)
    return mat


def collocation(N=64):
    return np.linspace(0, 1, N, endpoint=False) + 0.5 / N


class IntegralEquation:
    """ # noqa
    Initialize an instance of the IntegralEquation class.
    """

    def __init__(self, linear, f, lamb, a, b, K, Phi):  # noqa
        if linear:
            self.linear = True
        else:
            self.linear = False
        self.f = f
        self.lamb = lamb
        self.a = a
        self.b = b
        self.K = K
        self.Phi = Phi

    def solve(self):
        if self.linear:
            return self.solve_linear()
        else:
            raise NotImplementedError(
                'Nonlinear integral equations are not implemented yet.')

    def solve_linear(self):
        f = self.f
        lamb = self.lamb
        a = self.a
        b = self.b
        K = self.K
        Phi = self.Phi

        S_1 = np.zeros(N)
        for j in range(N):
            for k in range(N):
                S_1[j] += K(0, t[k]) * haar_int_1(t[k], j + 1)
        S_1 = 1 / N * S_1

        S_2 = 0
        for k in range(N):
            S_2 += K(0, t[k])
        S_2 = 1 / N * S_2

        M_A = np.zeros((N, N))
        for j in range(N):
            for k in range(N):
                M_A[:, j] += K(x, t[k]) * haar_int_1(t[k], j + 1)
        M_A = haar_int_1_mat(x, N) - 1 / N * M_A

        V_B = np.zeros(N)
        for k in range(N):
            V_B += K(x, t[k])
        V_B = 1 - 1 / N * V_B

        A_ls = M_A + np.outer(V_B, S_1) / (1 - S_2)
        B_ls = f(x) - f(0) * V_B / (1 - S_2)

        coef_haar = np.linalg.solve(A_ls, B_ls)

        return coef_haar