#!/usr/bin/env python

import numpy as np

def downsample(x, M):
    M = int(M)
    return x[::M]

def upsample(x, M):
    M = int(M)
    y = np.zeros(shape = (M * len(x), ), dtype = x.dtype)
    y[ : : M] = x
    return y

def signals_sum(x, y):
    if len(x) == len(y):
        return x + y
    if len(x) > len(y):
        y_ = np.zeros(shape = x.shape, dtype = y.dtype)
        y_[: len(y)] = y
        return x + y_
    return signals_sum(y, x)
    # x_ = np.zeros(shape = y.shape, dtype = x.dtype)
    # x_[: len(x)] = x
    # return x_ + y
