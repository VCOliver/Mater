#!/usr/bin/env python

# map <leader><leader> :wall<cr>:!python %<cr>

from basic_signal_operations_qmf import downsample, upsample, signals_sum
from matplotlib import pyplot as plt
import numpy as np
from qmf_perfect_reconstruction_eval import qmf_perfect_reconstruction_eval
from scipy.io import loadmat
from scipy.io.wavfile import read as wavread
from wfilters import wfilters

def qmf_analysis(h0, h1, x):
    x0 = np.convolve(h0, x)
    x1 = np.convolve(h1, x)
    x0 = downsample(x0, 2)
    x1 = downsample(x1, 2)
    xhat = np.zeros(shape = (len(x0) + len(x1), ), dtype = x.dtype)
    xhat[: len(x0)] = x0
    xhat[len(x0): ] = x1
    return xhat, x0, x1

def qmf_synthesis(x0, x1, g0, g1):
    x0 = upsample(x0, 2)
    x1 = upsample(x1, 2)
    x0 = np.convolve(g0, x0)
    x1 = np.convolve(g1, x1)
    return signals_sum(x0, x1)

if __name__ == '__main__':
    h0, h1, g0, g1 = wfilters('haar')
    h0, h1, g0, g1 = wfilters('db5')
    D = loadmat('ECG_1.mat')
    x = D['x']
    x = x[:, 0]
    x = x - np.mean(x)
    N = len(x)
    n = np.arange(0, len(x))
    plt.plot(n, x)
    plt.grid
    plt.title('Sinal de ECG Original')
    plt.xlabel('Índice de tempo n')
    plt.ylabel('x[n]')
    plt.show()
    x_hat, x0, x1 = qmf_analysis(h0, h1, x)
    x0_hat, x00, x01 = qmf_analysis(h0, h1, x0)
    x00_hat, x000, x001 = qmf_analysis(h0, h1, x00)
    x_hat3 = np.zeros(shape = (len(x000) + len(x001) + len(x01) + len(x1), ))
    x_hat3[: len(x000)] = x000
    x_hat3[len(x000) : len(x000) + len(x001)] = x001
    x_hat3[len(x000) + len(x001) : len(x000) + len(x001) + len(x01)] = x01
    x_hat3[len(x000) + len(x001) + len(x01) : ] = x1
    n = np.arange(0, len(x_hat3))
    plt.plot(n, x_hat3)
    plt.grid
    plt.xlabel('k')
    plt.ylabel('x_hat[k]')
    plt.title('Transformada Daubechies 5 do sinal de ECG com 3 níveis de decomposição')
    plt.show()
    x_hat = np.fft.fft(x)
    k = np.arange(0, len(x_hat))
    i = int(np.floor(len(x_hat) / 2))
    plt.plot(k[: i], np.abs(x_hat[: i]))
    plt.show()
