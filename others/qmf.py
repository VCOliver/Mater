#!/usr/bin/env python

# map <leader><leader> :wall<cr>:!python qmf.py<cr>

from basic_signal_operations_qmf import downsample, upsample, signals_sum
from copy import deepcopy
from matplotlib import pyplot as plt
import numpy as np
from qmf_perfect_reconstruction_eval import qmf_perfect_reconstruction_eval
from scipy.io import loadmat
from scipy.signal import freqz
from wfilters import wfilters
# import wfilters

def font_configuration():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Palatino"],
        "font.size": 25,
    })

def qmf_filter_iterator(h0, h1, levels):
    conv = np.convolve
    h = [h1] * (levels + 1);
    h1_iterated = h1
    h0_iterated = h0
    for k in range(0, levels - 1):
        h1_iterated = conv(h0, upsample(h1_iterated, 2)) 
        h0_iterated = conv(h0, upsample(h0_iterated, 2)) 
        h[len(h) - 2 - k] = h1_iterated
    h[0] = h0_iterated
    multirate_factors = np.zeros(shape = (levels + 1, ))
    multirate_factors[1 : levels + 1] = \
    2.0 ** np.arange(levels, 0, -1)
    multirate_factors[0] = 2.0 ** levels 
    return h, multirate_factors

def spectra_subbands(h):
    for k in range(0, len(h)):
        w, H = freqz(h[k], 10000)
        plt.plot(w / (2 * np.pi), abs(H))
    plt.show()

def qmf_decomposition(h, multirate_factors, x):
    conv = np.convolve
    x_qmf = [x] * len(h)
    L = 0
    for k in range(0, len(h)):
        x_qmf[k] = conv(h[k], x_qmf[k])
        x_qmf[k] = downsample(x_qmf[k], int(multirate_factors[k]))
        L += len(x_qmf[k])
    x_hat = np.zeros(shape = (L, ))
    n = 0
    for k in range(0, len(h)):
        x_ = x_qmf[k]
        x_hat[n : n + len(x_),] = x_
        n += len(x_)
    return x_qmf, x_hat

def compensation_blocks(A, d, levels):
    factors = np.zeros(shape = (levels + 1, ))
    factors[0] = A ** (-levels)
    factors[1:] = A ** np.arange(-levels, 0, +1)
    shifts = np.zeros(shape = (levels + 1, ))
    shifts[0] = (2 ** levels) - 1
    shifts[1:] = 2 ** np.arange(levels, 0, -1) - 1
    shifts *= d
    return factors, shifts

def qmf_reconstruction(g, multirate_factors, factors, shifts, x_qmf, N):
    conv = np.convolve
    xr = np.zeros(shape = (1, ))
    for k in range(0, len(x_qmf)):
        y = upsample(x_qmf[k], multirate_factors[k])
        y *= factors[k]
        y = conv(g[k], y)
        y = y[int(shifts[k]) :]
        xr = signals_sum(xr, y)
    return xr[:N]

if __name__ == '__main__':
    font_configuration()
    h0, h1, g0, g1 = wfilters('db5')
    levels = 10
    h, multirate_factors = qmf_filter_iterator(h0, h1, levels)
    g, _ = qmf_filter_iterator(g0, g1, levels)
    _, A, d = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
    factors, shifts = compensation_blocks(A, d, levels)
    print(factors)
    print(shifts / d)
    D = loadmat('./ECG_1.mat')
    x = D['x'][:, 0]
    # x = x[: 1000]
    fs = D['fs']
    x_qmf, x_hat = qmf_decomposition(h, multirate_factors, x)
    k = np.arange(0, len(x_hat))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(k) - np.min(k)
    yrange = np.max(x_hat) - np.min(x_hat)
    # ax.set_aspect(5.0 * xrange / yrange / 8.0)
    # plt.tight_layout()
    plt.plot(k, x_hat)
    plt.xlabel(r'Índice da transformada (${k}$)')
    plt.ylabel(r'${\hat{x}[k]}$')
    plt.grid()
    plt.show()
    xr = qmf_reconstruction(g, multirate_factors, factors, shifts, x_qmf, len(x))
    n = np.arange(0, len(xr))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(n) - np.min(n)
    yrange = np.max(xr) - np.min(xr)
    # ax.set_aspect(5.0 * xrange / yrange / 8.0)
    # plt.tight_layout()
    plt.plot(n, xr, label = 'sinal reconstruído')
    n = np.arange(0, len(x))
    plt.plot(n, x, 'r--', label = 'sinal original')
    plt.xlabel('Índice de tempo n')
    plt.ylabel('${x_r[n]}$', rotation = 0.0, labelpad = 20)
    plt.grid()
    plt.legend(fontsize=12.0)
    plt.show()
    # Sinal de erro
    fig = plt.figure()
    ax = fig.add_subplot(111)
    xrange = np.max(n) - np.min(n)
    yrange = np.max(xr - x) - np.min(xr - x)
    ax.set_aspect(5.0 * xrange / yrange / 8.0) # or: forceAspect(ax)
    plt.tight_layout()
    plt.plot(n, xr - x)
    plt.xlabel('Índice de tempo n')
    plt.ylabel('${E[n]}$', rotation = 0.0, labelpad = 20)
    plt.grid()
    plt.legend(fontsize=12.0)
    plt.show()
