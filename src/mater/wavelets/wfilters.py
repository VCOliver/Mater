"""
Wavelet filter coefficient loader.

Loads decomposition and reconstruction filter coefficients (h0, h1, g0, g1)
for named wavelet families from a bundled .mat data file.

MATLAB equivalent: [Lo_D, Hi_D, Lo_R, Hi_R] = wfilters('db5')
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat


# Path to the bundled wfilters.mat data file
_DATA_DIR = Path(__file__).parent / "data"
_WFILTERS_MAT = _DATA_DIR / "wfilters.mat"


def wfilters(wname: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load wavelet filter coefficients for a given wavelet name.

    Reads the four filter coefficient arrays (h0, h1, g0, g1) from the
    bundled ``wfilters.mat`` file. These correspond to the low-pass and
    high-pass decomposition/reconstruction filters used in discrete
    wavelet transforms.

    Parameters
    ----------
    wname : str
        Wavelet name (e.g. ``'db5'``, ``'sym8'``, ``'coif3'``).
        Dots in the name are replaced with underscores to match
        the variable naming convention inside the .mat file.

    Returns
    -------
    h0 : np.ndarray
        Low-pass decomposition filter coefficients.
    h1 : np.ndarray
        High-pass decomposition filter coefficients.
    g0 : np.ndarray
        Low-pass reconstruction filter coefficients.
    g1 : np.ndarray
        High-pass reconstruction filter coefficients.

    Examples
    --------
    >>> from mater.wavelets import wfilters
    >>> h0, h1, g0, g1 = wfilters('db5')
    """
    # Replace dots with underscores to match .mat variable naming convention
    wname = wname.replace(".", "_")

    data = loadmat(str(_WFILTERS_MAT))

    h0 = data["h0_" + wname][0]
    h1 = data["h1_" + wname][0]
    g0 = data["g0_" + wname][0]
    g1 = data["g1_" + wname][0]

    return h0, h1, g0, g1
