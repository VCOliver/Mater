"""
QMF (Quadrature Mirror Filter) analysis, synthesis, and evaluation.

Provides the building blocks for two-channel QMF filter-bank processing:

- **Single-level analysis/synthesis** -- decompose a signal into subbands
  and reconstruct it back.
- **Multi-level (iterated) filter bank** -- build equivalent filters for
  an N-level wavelet decomposition tree, decompose/reconstruct through
  all levels at once, and compute the per-subband gain/delay compensation
  needed for perfect reconstruction.
- **Perfect-reconstruction evaluation** -- verify that a set of four
  analysis/synthesis filters satisfies the aliasing-cancellation and
  distortion-free conditions.
"""

from copy import deepcopy

import numpy as np

from mater.wavelets.signal_ops import downsample, upsample, signals_sum


def qmf_analysis(
    h0: np.ndarray,
    h1: np.ndarray,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Single-level QMF analysis (decomposition) of a signal.

    Filters the input signal ``x`` through the low-pass (``h0``) and
    high-pass (``h1``) analysis filters, then downsamples each branch
    by a factor of 2 to produce the two subband signals.

    .. code-block:: text

                  ┌──[h0]──↓2──► x0  (low-pass / approximation)
        x ───────┤
                  └──[h1]──↓2──► x1  (high-pass / detail)

    Parameters
    ----------
    h0 : np.ndarray
        Low-pass analysis (decomposition) filter coefficients.
    h1 : np.ndarray
        High-pass analysis (decomposition) filter coefficients.
    x : np.ndarray
        Input signal to decompose.

    Returns
    -------
    x0 : np.ndarray
        Low-pass (approximation) subband coefficients.
    x1 : np.ndarray
        High-pass (detail) subband coefficients.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import wfilters, qmf_analysis
    >>> h0, h1, g0, g1 = wfilters('db5')
    >>> x = np.random.randn(100)
    >>> x0, x1 = qmf_analysis(h0, h1, x)
    """
    # Filter through both analysis branches
    x0 = np.convolve(h0, x)
    x1 = np.convolve(h1, x)

    # Critically-sampled: keep every other sample
    x0 = downsample(x0, 2)
    x1 = downsample(x1, 2)

    return x0, x1


def qmf_synthesis(
    x0: np.ndarray,
    x1: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
) -> np.ndarray:
    """
    Single-level QMF synthesis (reconstruction) from two subbands.

    Upsamples each subband by 2, filters through the corresponding
    reconstruction filter, and sums the two branches to recover the
    (possibly delayed) original signal.

    .. code-block:: text

        x0 ──↑2──[g0]──┐
                        ├──(+)──► y  (reconstructed signal)
        x1 ──↑2──[g1]──┘

    Parameters
    ----------
    x0 : np.ndarray
        Low-pass (approximation) subband coefficients.
    x1 : np.ndarray
        High-pass (detail) subband coefficients.
    g0 : np.ndarray
        Low-pass synthesis (reconstruction) filter coefficients.
    g1 : np.ndarray
        High-pass synthesis (reconstruction) filter coefficients.

    Returns
    -------
    y : np.ndarray
        Reconstructed signal. If the filters satisfy perfect
        reconstruction, ``y`` equals the original signal scaled
        by a constant gain and shifted by a fixed delay.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import wfilters, qmf_analysis, qmf_synthesis
    >>> h0, h1, g0, g1 = wfilters('haar')
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> x0, x1 = qmf_analysis(h0, h1, x)
    >>> y = qmf_synthesis(x0, x1, g0, g1)
    """
    # Restore the original sampling rate by inserting zeros
    x0_up = upsample(x0, 2)
    x1_up = upsample(x1, 2)

    # Filter through the synthesis branches
    y0 = np.convolve(g0, x0_up)
    y1 = np.convolve(g1, x1_up)

    # Combine both branches (zero-pads if lengths differ)
    return signals_sum(y0, y1)


def qmf_filter_iterator(
    h0: np.ndarray,
    h1: np.ndarray,
    levels: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Build the equivalent analysis filters for an N-level QMF tree.

    Starting from the two prototype filters (``h0``, ``h1``), this
    function iteratively upsamples and convolves to produce the set of
    ``levels + 1`` equivalent filters that, when applied directly to the
    input signal, give the same subband coefficients as cascading
    single-level decompositions.

    The returned ``multirate_factors`` array gives the total downsampling
    factor for each subband (all powers of 2).

    .. code-block:: text

        Level 1:  h1                              (detail, ↓2)
        Level 2:  conv(h0, ↑2{h1})               (detail, ↓4)
        ...
        Level L:  conv(h0, ↑2{...conv(h0, ↑2{h1})})  (detail, ↓2^L)
        Residual: conv(h0, ↑2{...conv(h0, ↑2{h0})})  (approx, ↓2^L)

    Parameters
    ----------
    h0 : np.ndarray
        Low-pass prototype filter (analysis or synthesis).
    h1 : np.ndarray
        High-pass prototype filter (analysis or synthesis).
    levels : int
        Number of decomposition levels (must be >= 1).

    Returns
    -------
    h : list[np.ndarray]
        List of ``levels + 1`` equivalent filters, ordered as:
        ``[approx, detail_L, detail_{L-1}, ..., detail_1]``.
    multirate_factors : np.ndarray
        Integer downsampling factor for each subband, same order as ``h``.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import wfilters, qmf_filter_iterator
    >>> h0, h1, g0, g1 = wfilters('db5')
    >>> h, mf = qmf_filter_iterator(h0, h1, levels=3)
    >>> len(h)       # 3 detail subbands + 1 approximation
    4
    >>> mf           # downsampling factors: [8, 8, 4, 2]
    array([8, 8, 4, 2])
    """
    # Start with the prototype high-pass at every slot; we overwrite
    # from the deepest level upward, leaving h[-1] = h1 (level-1 detail).
    h: list[np.ndarray] = [h1] * (levels + 1)

    h1_iterated = h1
    h0_iterated = h0

    for k in range(levels - 1):
        # Upsample the previous iterated filter by 2, then convolve
        # with h0 to push one level deeper into the tree.
        h1_iterated = np.convolve(h0, upsample(h1_iterated, 2))
        h0_iterated = np.convolve(h0, upsample(h0_iterated, 2))
        # Fill from the second-to-last slot backwards
        h[len(h) - 2 - k] = h1_iterated

    # Slot 0 is the deepest low-pass (approximation) branch
    h[0] = h0_iterated

    # Downsampling factors: 2^levels for approx and deepest detail,
    # then 2^(levels-1), 2^(levels-2), ..., 2^1 for shallower details.
    multirate_factors = np.empty(levels + 1, dtype=int)
    multirate_factors[0] = 1 << levels
    multirate_factors[1:] = [1 << k for k in range(levels, 0, -1)]

    return h, multirate_factors


def qmf_decomposition(
    h: list[np.ndarray],
    multirate_factors: np.ndarray,
    x: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray]:
    """
    Multi-level QMF decomposition using pre-built iterated filters.

    Convolves the input signal with each equivalent filter from
    :func:`qmf_filter_iterator` and downsamples by the corresponding
    factor to produce the subband coefficients. Also returns all
    subbands concatenated into a single coefficient vector.

    Parameters
    ----------
    h : list[np.ndarray]
        Iterated filter bank from :func:`qmf_filter_iterator`.
    multirate_factors : np.ndarray
        Per-subband downsampling factors from :func:`qmf_filter_iterator`.
    x : np.ndarray
        Input signal to decompose.

    Returns
    -------
    x_qmf : list[np.ndarray]
        List of subband coefficient arrays, same order as ``h``.
    x_hat : np.ndarray
        All subband coefficients concatenated into one vector
        (useful for plotting the full wavelet transform).

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import wfilters, qmf_filter_iterator, qmf_decomposition
    >>> h0, h1, g0, g1 = wfilters('db5')
    >>> h, mf = qmf_filter_iterator(h0, h1, levels=3)
    >>> x = np.random.randn(256)
    >>> x_qmf, x_hat = qmf_decomposition(h, mf, x)
    """
    x_qmf: list[np.ndarray] = []
    total_length = 0

    for k in range(len(h)):
        # Filter and downsample each subband independently
        coeffs = np.convolve(h[k], x)
        coeffs = downsample(coeffs, multirate_factors[k])
        x_qmf.append(coeffs)
        total_length += len(coeffs)

    # Concatenate all subbands into a single coefficient vector
    x_hat = np.empty(total_length)
    offset = 0
    for coeffs in x_qmf:
        x_hat[offset : offset + len(coeffs)] = coeffs
        offset += len(coeffs)

    return x_qmf, x_hat


def compensation_blocks(
    A: float,
    d: int,
    levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-subband gain and delay compensation for reconstruction.

    When the prototype filters have gain ``A`` and delay ``d`` (as
    returned by :func:`qmf_perfect_reconstruction_eval`), the multi-level
    filter bank accumulates different gains and delays at each level.
    This function computes the correction factors needed to undo that
    accumulation during reconstruction.

    Parameters
    ----------
    A : float
        Single-level transfer-function gain from
        :func:`qmf_perfect_reconstruction_eval`.
    d : int
        Single-level sample delay from
        :func:`qmf_perfect_reconstruction_eval`.
    levels : int
        Number of decomposition levels.

    Returns
    -------
    factors : np.ndarray
        Multiplicative gain correction for each subband. Ordered as
        ``[approx, detail_L, detail_{L-1}, ..., detail_1]``.
    shifts : np.ndarray
        Integer sample shifts to trim from each reconstructed subband
        to align it in time with the original signal.

    Examples
    --------
    >>> from mater.wavelets import compensation_blocks
    >>> factors, shifts = compensation_blocks(A=1.0, d=9, levels=3)
    """
    # Gain correction: approx branch passes through all L stages,
    # detail at level k passes through (L - k + 1) stages.
    factors = np.empty(levels + 1)
    factors[0] = A ** (-levels)
    factors[1:] = [A ** k for k in range(-levels, 0)]

    # Delay compensation: each level's upsampling doubles the
    # accumulated delay; use bit-shift for the powers of 2.
    shifts = np.empty(levels + 1, dtype=int)
    shifts[0] = ((1 << levels) - 1) * d
    shifts[1:] = [((1 << k) - 1) * d for k in range(levels, 0, -1)]

    return factors, shifts


def qmf_reconstruction(
    g: list[np.ndarray],
    multirate_factors: np.ndarray,
    factors: np.ndarray,
    shifts: np.ndarray,
    x_qmf: list[np.ndarray],
    N: int,
) -> np.ndarray:
    """
    Multi-level QMF reconstruction from subband coefficients.

    Reverses :func:`qmf_decomposition` by upsampling each subband,
    applying gain compensation, filtering through the iterated
    synthesis filters, trimming the accumulated delay, and summing
    all branches.

    Parameters
    ----------
    g : list[np.ndarray]
        Iterated synthesis filter bank (from :func:`qmf_filter_iterator`
        called with ``g0, g1``).
    multirate_factors : np.ndarray
        Per-subband upsampling factors (same array used for decomposition).
    factors : np.ndarray
        Per-subband gain corrections from :func:`compensation_blocks`.
    shifts : np.ndarray
        Per-subband delay corrections from :func:`compensation_blocks`.
    x_qmf : list[np.ndarray]
        Subband coefficients from :func:`qmf_decomposition`.
    N : int
        Length of the original signal (output is trimmed to this length).

    Returns
    -------
    xr : np.ndarray
        Reconstructed signal of length ``N``.

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import (
    ...     wfilters, qmf_filter_iterator, qmf_decomposition,
    ...     qmf_perfect_reconstruction_eval, compensation_blocks,
    ...     qmf_reconstruction,
    ... )
    >>> h0, h1, g0, g1 = wfilters('db5')
    >>> levels = 3
    >>> h, mf = qmf_filter_iterator(h0, h1, levels)
    >>> g, _  = qmf_filter_iterator(g0, g1, levels)
    >>> _, A, d = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
    >>> fct, sh = compensation_blocks(A, d, levels)
    >>> x = np.random.randn(256)
    >>> x_qmf, _ = qmf_decomposition(h, mf, x)
    >>> xr = qmf_reconstruction(g, mf, fct, sh, x_qmf, len(x))
    >>> np.allclose(x, xr)
    True
    """
    xr = np.zeros(1)

    for k in range(len(x_qmf)):
        # Upsample back to the original rate
        y = upsample(x_qmf[k], multirate_factors[k])

        # Apply per-subband gain correction
        y *= factors[k]

        # Filter through the corresponding synthesis filter
        y = np.convolve(g[k], y)

        # Trim the accumulated delay for this subband
        y = y[shifts[k]:]

        # Accumulate into the output (zero-pads if lengths differ)
        xr = signals_sum(xr, y)

    # Return only the first N samples (original signal length)
    return xr[:N]


def qmf_perfect_reconstruction_eval(
    h0: np.ndarray,
    h1: np.ndarray,
    g0: np.ndarray,
    g1: np.ndarray,
    tol: float = 1e-4,
) -> tuple[bool, float, int]:
    """
    Evaluate whether a QMF filter bank achieves perfect reconstruction.

    The function checks two conditions for the analysis filters
    (h0, h1) and synthesis filters (g0, g1):

    1. **Aliasing cancellation** -- the aliased components introduced by
       downsampling must cancel when the subbands are recombined.
    2. **LTI (distortion-free) condition** -- the combined transfer
       function must be a pure delay scaled by a constant gain.

    Parameters
    ----------
    h0 : np.ndarray
        Low-pass analysis (decomposition) filter.
    h1 : np.ndarray
        High-pass analysis (decomposition) filter.
    g0 : np.ndarray
        Low-pass synthesis (reconstruction) filter.
    g1 : np.ndarray
        High-pass synthesis (reconstruction) filter.
    tol : float, optional
        Numerical tolerance for the zero checks. Default is ``1e-4``.

    Returns
    -------
    is_pr : bool
        ``True`` if the filter bank satisfies perfect reconstruction
        within the given tolerance.
    gain : float
        The gain constant *A* of the overall transfer function.
        Zero when perfect reconstruction fails.
    delay : int
        The sample delay *d* introduced by the filter bank.
        Negative error codes when perfect reconstruction fails:

        * ``-2`` -- aliasing cancellation failed.
        * ``-4`` -- gain is effectively zero (degenerate system).
        * ``-5`` -- LTI condition failed (non-trivial distortion).

    Examples
    --------
    >>> import numpy as np
    >>> from mater.wavelets import qmf_perfect_reconstruction_eval
    >>> h0 = np.array([1, 1, 0]) / np.sqrt(2)
    >>> h1 = np.array([1, -1, 0, 0, 0]) / np.sqrt(2)
    >>> g0 = np.array([1, 1, 0, 0, 0, 0, 0]) / np.sqrt(2)
    >>> g1 = np.array([-1, 1]) / np.sqrt(2)
    >>> is_pr, gain, delay = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
    """
    # --- Aliasing cancellation check ---
    # Modulate the analysis filters by (-1)^n to shift them by pi
    h0_mod = deepcopy(h0)
    h0_mod[1::2] = -h0_mod[1::2]

    h1_mod = deepcopy(h1)
    h1_mod[1::2] = -h1_mod[1::2]

    # Aliasing term: H0(-z)G0(z) + H1(-z)G1(z) should be zero
    h0_g0_alias = np.convolve(h0_mod, g0)
    h1_g1_alias = np.convolve(h1_mod, g1)
    aliasing_term = signals_sum(h0_g0_alias, h1_g1_alias)

    if np.any(np.abs(aliasing_term) > tol):
        return False, 0, -2

    # --- LTI (distortion) condition check ---
    # Transfer function: 0.5 * [H0(z)G0(z) + H1(z)G1(z)] should be A * z^{-d}
    h0g0 = np.convolve(h0, g0)
    h1g1 = np.convolve(h1, g1)
    lti_term = 0.5 * signals_sum(h0g0, h1g1)

    # Find the dominant coefficient (gain A at delay d)
    d = np.argmax(np.abs(lti_term))
    A = lti_term[d]

    if np.abs(A) <= tol:
        return False, 0, -4

    # All other coefficients should be zero (pure delay)
    lti_term[d] = 0
    if np.any(np.abs(lti_term) > tol):
        return False, 0, -5

    return True, A, d
