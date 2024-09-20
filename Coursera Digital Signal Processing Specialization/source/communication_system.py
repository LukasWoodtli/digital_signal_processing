import numpy as np
from commpy import rcosfilter
from matplotlib import pyplot as plt


def test_raised_cosine_filter():
    N = 100
    beta = 0.75  # in commpy it's called alpha
    T_s = 1/100.
    F_s = 1000
    t, imp_resp = rcosfilter(N, beta, T_s, F_s)

    plt.stem(t, imp_resp)
    #plt.show()

