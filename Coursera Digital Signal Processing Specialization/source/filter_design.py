import os.path
from pathlib import Path

from matplotlib.gridspec import GridSpec
from scipy.signal import bode, dlti, TransferFunction, dfreqresp, tf2zpk
import scipy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as patches


def test_leaky_integrator():
    # leaky integrator
    # $$H(z) = /frac{(1-\lambda)}{1 - \lambda z^{-1}}
    # = /frac{(1-\lambda) z}{1 z - \lambda}$$
    lmbd = 0.98  # lambda
    b = [1 - lmbd]
    a = [1, -lmbd]

    dlti = scipy.signal.dlti(b, a)
    plot_dlti(dlti, "Leaky integrator")


def plot_dlti(dlti, name):
    all_fig = plt.figure(layout="constrained")
    all_grid_spec = GridSpec(2, 2, figure=all_fig)

    # frequency response
    w = np.linspace(-np.pi, np.pi, 100)
    w2, h2 = dlti.freqresp(w=w)
    freq_resp_grid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=all_grid_spec[0])
    ax1 = all_fig.add_subplot(freq_resp_grid_spec[0, 0])
    ax1.set(title='Frequency Response')
    ax1.plot(w2, np.abs(h2))
    ax1.grid(True)
    ax2 = all_fig.add_subplot(freq_resp_grid_spec[1, 0])
    ax2.plot(w2, np.angle(h2))
    ax2.grid(True)

    # poles-zeroes plot
    # inspired by:
    # https://github.com/bmcfee/dstbook/blob/main/content/ch12-ztransform/PoleZero.ipynb
    zpg = scipy.signal.ZerosPolesGain(dlti)
    (zeros, poles, gain) = (zpg.zeros, zpg.poles, zpg.gain)

    pole_zero_grid_spec = gridspec.GridSpecFromSubplotSpec(1, 1, subplot_spec=all_grid_spec[1])

    ax = all_fig.add_subplot(pole_zero_grid_spec[0, 0])

    circ = patches.Ellipse((0, 0), 2, 2, edgecolor='k', linewidth=3, fill=False, zorder=-10)

    ax.add_patch(circ)
    ax.set(xlim=[-1.6, 1.6], ylim=[-1.6, 1.6], aspect='equal', xlabel='Real', ylabel='Imaginary')

    ax.scatter(poles.real, poles.imag, marker='x', s=100, color='r', label='Poles')
    ax.scatter(zeros.real, zeros.imag, marker='o', s=100, edgecolor='b', color=None, facecolor='none', label='Zeros')
    ax.set(title='Poles plot')



    # bode
    w, mag, phase = scipy.signal.dbode(dlti)

    bode_grid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=all_grid_spec[1, 0])
    ax1 = all_fig.add_subplot(bode_grid_spec[0, 0])
    ax1.set(title='Bode Plot')
    ax1.semilogx(w, mag)  # Bode magnitude plot
    ax1.set_ylabel('Magnitude (dB)')
    ax1.grid(True)
    ax2 = all_fig.add_subplot(bode_grid_spec[1, 0])
    ax2.semilogx(w, phase)  # Bode phase plot
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True)

    # Time domain
    # impulse response
    t, y = scipy.signal.dimpulse(dlti)
    time_domain_grid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=all_grid_spec[1, 1])
    ax1 = all_fig.add_subplot(time_domain_grid_spec[0, 0])
    ax1.set(title='Time Domain')
    ax1.step(t, np.squeeze(y))
    ax1.set_xlabel('n [samples]')
    ax1.set_ylabel('Amplitude\nimpulse resp.')
    ax1.grid(True)
    # step response
    t, y = scipy.signal.dstep(dlti)
    ax2 = all_fig.add_subplot(time_domain_grid_spec[1, 0])
    ax2.step(t, np.squeeze(y))
    ax2.set_xlabel('n [samples]')
    ax2.set_ylabel('Amplitude\nstep resp.')
    ax2.grid(True)

    #plt.show()

    dir_path = Path(__file__).parent.absolute() / "filter_design_plots"
    os.makedirs(dir_path, exist_ok=True)
    file_path = dir_path / f"{name}.svg"
    plt.savefig(file_path)
