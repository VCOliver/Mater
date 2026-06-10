"""
Wavelets subpackage for the mater library.

Provides wavelet filter loading, basic multirate signal operations,
and QMF analysis/synthesis/evaluation -- bringing MATLAB-style
wavelet toolbox functionality to Python.

Modules
-------
wfilters
    Load wavelet filter coefficients by name.
signal_ops
    Downsample, upsample, and length-safe signal addition.
qmf
    QMF analysis, synthesis, and perfect reconstruction evaluation
    for two-channel filter banks.
"""

from mater.wavelets.wfilters import wfilters
from mater.wavelets.signal_ops import downsample, upsample, signals_sum
from mater.wavelets.qmf import (
    qmf_analysis,
    qmf_synthesis,
    qmf_filter_iterator,
    qmf_decomposition,
    compensation_blocks,
    qmf_reconstruction,
    qmf_perfect_reconstruction_eval,
)

__all__ = [
    "wfilters",
    "downsample",
    "upsample",
    "signals_sum",
    "qmf_analysis",
    "qmf_synthesis",
    "qmf_filter_iterator",
    "qmf_decomposition",
    "compensation_blocks",
    "qmf_reconstruction",
    "qmf_perfect_reconstruction_eval",
]
