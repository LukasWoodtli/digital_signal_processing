import matplotlib.pyplot as plt
import numpy as np

from source.dft import calc_dtf


def plot_dft_fft():
    N = 128

    x = np.zeros(N)
    x[0:64] = 1

    # plot signal (step function)
    plt.stem(x)
    plt.show()

    # use custom DFT function
    X = calc_dtf(x)
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.stem(abs(X))
    ax2.stem(np.angle(X))
    plt.show()

    # use FFT from numpy (for better result)
    X = np.fft.fft(x)
    f, (ax1, ax2) = plt.subplots(2, 1)
    ax1.stem(abs(X))
    ax2.stem(np.angle(X))
    plt.show()
