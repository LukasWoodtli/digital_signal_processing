import numpy as np
import sympy.abc
from matplotlib import pyplot as plt


def test_complex_exponential():
    res = sympy.exp(-sympy.I * sympy.pi)
    assert res == -1


def dtf_matrix(N):
    # W_N(n, k) = e^(-j * 2pi/N * n * k)
    a = np.arange(N)
    a = np.expand_dims(a, 0)

    W = np.exp(-2j * (np.pi / N) * a.T * a)

    return W


def calc_dtf(x):
    N = len(x)
    W = dtf_matrix(N)
    X = np.dot(W, x)
    return X

def inv_dtf_matrix(N):
    W = dtf_matrix(N)
    W = W.T.conjugate() / N
    return W


def calc_inv_dtf(X):
    N = len(X)
    W_inf = inv_dtf_matrix(N)
    x = np.dot(W_inf, X)
    return x


def test_dtf_matrix():
    x = np.array([5, 7, 9])

    # DTF
    X = calc_dtf(x)
    np.testing.assert_array_almost_equal(X,
                                         [21.+0.j, -3.+1.732051j, -3.-1.732051j])

    # inverse DTF
    x_hat = calc_inv_dtf(X)

    np.testing.assert_array_almost_equal(x, x_hat)


def plot_dft_coefficients():
    # See 4.2 The DFT (Discrete Fourier Transform)

    N = 32

    def W(N):
        return np.exp(-2j * (np.pi / N))

    def w(k):
        x = np.zeros(N, dtype=np.complex_)
        x[0] = 1
        for i in range(1, N):
            x[i] = W(N)**(-i*k)
        return x.T

    def plot_coefficients(w):
        f, (ax1, ax2) = plt.subplots(2, 1)
        ax1.stem(w.real)
        ax1.set_ylabel('Re')
        ax2.stem(w.imag)
        ax2.set_ylabel('Im')
        plt.show()

    # See also book
    w_0 = w(0)
    plot_coefficients(w_0)
    w_1 = w(1)
    plot_coefficients(w_1)
    w_7 = w(7)
    plot_coefficients(w_7)
    w_31 = w(31)
    plot_coefficients(w_31)


