import numpy
import numpy as np
from matplotlib import pyplot as plt
from sympy.physics.quantum.identitysearch import scipy


def test_convolution_0():
    #  h[n] = delta[n] - delta[n - 1]
    #  x[n] = 1, n >= 0; 0 else
    n = np.arange(0, 10, 1)
    l = len(n)
    h = scipy.signal.unit_impulse(l, 0) - scipy.signal.unit_impulse(l, 1)
    x = np.ones(l)

    y = np.convolve(h, x)
    y = np.concatenate(([0], y))  # shift to right to include n=-1
    np.testing.assert_almost_equal(y[:4], [0, 1, 0, 0])


def test_convolution_0():
    #  h[n] = delta[n] - delta[n - 1]
    #  x[n] = n, n >= 0; 0 else
    n = np.arange(0, 10, 1)
    l = len(n)
    h = scipy.signal.unit_impulse(l, 0) - scipy.signal.unit_impulse(l, 1)
    x = np.array([n for n in range(l)])

    #plt.stem(x)
    #plt.stem(h)
    #plt.show()
    y = np.convolve(h, x)
    y = np.concatenate(([0], y))  # shift to right to include n=-1
    np.testing.assert_almost_equal(y[:4], [0, 0, 1, 1])


def test_convolution_1():
    #  x[n] = cos(pi/2*n)
    #  h[n] = 1/5 * sinc(n/5)
    n = np.linspace(0, 9, 10)
    x = np.cos(np.pi / 2 * n)
    h = 1/5 * np.sinc(n / 5)

    #plt.stem(h)
    #plt.show()
    y = np.convolve(h, x)
    np.testing.assert_almost_equal(y[5], 0.08618762)

