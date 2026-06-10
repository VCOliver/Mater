#!/usr/bin/env python

import numpy as np
from scipy.io import loadmat

def wfilters(wname):
    wname.replace('.', '_')
    D = loadmat('wfilters.mat')
    h0 = D['h0_' + wname][0]
    h1 = D['h1_' + wname][0]
    g0 = D['g0_' + wname][0]
    g1 = D['g1_' + wname][0]
    return h0, h1, g0, g1

if __name__ == '__main__':
    h0, h1, g0, g1 = wfilters('db5')
    print(h0)
    print(h1)
    print(g0)
    print(g1)
