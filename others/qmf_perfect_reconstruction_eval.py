#!/usr/bin/env python

# map <leader><leader> :wall<cr>:!python %<cr>

from copy import deepcopy
import numpy as np
from basic_signal_operations_qmf import signals_sum
from wfilters import wfilters

def qmf_perfect_reconstruction_eval(h0, h1, g0, g1, tol = 1e-4):
    h0_ = deepcopy(h0)
    # h0_ = (h0 + [])[:-1]
    h0_[1 : : 2] = -h0_[1 : : 2]
    h1_ = deepcopy(h1)
    h1_[1 : : 2] = -h1_[1 : : 2]
    h0_g0 = np.convolve(h0_, g0)
    h1_g1 = np.convolve(h1_, g1)
    aliasing_term = signals_sum(h0_g0, h1_g1)
    if np.any(np.abs(aliasing_term) > tol):
        return False, 0, -2 
    h0g0 = np.convolve(h0, g0)
    h1g1 = np.convolve(h1, g1)
    lti_term = 0.5 * signals_sum(h0g0, h1g1)
    d = np.argmax(np.abs(lti_term))
    A = lti_term[d]
    if np.abs(A) <= tol:
        return False, 0, -4
    lti_term[d] = 0
    if np.any(np.abs(lti_term) > tol):
        return False, 0, -5 
    return True, A, d

if __name__ == '__main__':
    h0 = np.array([1, 1, 0]) / np.sqrt(2)
    h1 = np.array([1, -1, 0, 0, 0]) / np.sqrt(2)
    g0 = np.array([1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)
    g1 = np.array([-1, 1]) / np.sqrt(2)
    perfect_reconstruction, A, d = \
    qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
    print(perfect_reconstruction)
    print(A)
    print(d)
    '''
    h0, h1, g0, g1 = wfilters('db17')
    perfect_reconstruction, A, d = \
    qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
    print(perfect_reconstruction)
    print(A)
    print(d)
    print('Comprimentos dos filtros:')
    print(len(h0))
    print(len(h1))
    print(len(g0))
    print(len(g1))
    '''
