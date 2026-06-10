"""
Basic signal operations for QMF / wavelet processing.

Provides downsample, upsample, and length-safe signal addition utilities
commonly needed in multirate signal processing and filter-bank analysis.

MATLAB equivalents:
    - downsample(x, M)
    - upsample(x, M)
    - Addition of signals with mismatched lengths (zero-padded)
"""

import numpy as np


def downsample(x: np.ndarray, M: int) -> np.ndarray:
    """
    Downsample a signal by keeping every M-th sample.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    M : int
        Downsampling factor. Every M-th sample is retained.

    Returns
    -------
    np.ndarray
        Downsampled signal of length ``ceil(len(x) / M)``.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import downsample
    >>> downsample(np.array([1, 2, 3, 4, 5, 6]), 2)
    array([1, 3, 5])
    """
    M = int(M)
    return x[::M]


def upsample(x: np.ndarray, M: int) -> np.ndarray:
    """
    Upsample a signal by inserting M-1 zeros between each sample.

    Parameters
    ----------
    x : np.ndarray
        Input signal array.
    M : int
        Upsampling factor. M-1 zeros are inserted between samples.

    Returns
    -------
    np.ndarray
        Upsampled signal of length ``M * len(x)``.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import upsample
    >>> upsample(np.array([1, 2, 3]), 3)
    array([1, 0, 0, 2, 0, 0, 3, 0, 0])
    """
    M = int(M)
    y = np.zeros(shape=(M * len(x),), dtype=x.dtype)
    y[::M] = x
    return y


def signals_sum(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Add two signals, zero-padding the shorter one to match lengths.

    When ``x`` and ``y`` have different lengths, the shorter signal is
    padded with trailing zeros so that element-wise addition is valid.

    Parameters
    ----------
    x : np.ndarray
        First input signal.
    y : np.ndarray
        Second input signal.

    Returns
    -------
    np.ndarray
        Element-wise sum of the two (possibly zero-padded) signals.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import signals_sum
    >>> signals_sum(np.array([1, 2, 3]), np.array([10, 20]))
    array([11, 22,  3])
    """
    # Equal-length case: straight addition
    if len(x) == len(y):
        return x + y

    # Ensure x is the longer signal, then zero-pad y to match
    if len(x) > len(y):
        y_padded = np.zeros(shape=x.shape, dtype=y.dtype)
        y_padded[: len(y)] = y
        return x + y_padded

    # x is shorter -- recurse with swapped arguments
    return signals_sum(y, x)
