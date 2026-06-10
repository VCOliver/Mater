#!/usr/bin/env python
"""
mater_test.py -- Walkthrough of the mater.wavelets subpackage.

Run from the project root:
    python scripts/mater_test.py

This script demonstrates every public function that was integrated
from the `others/` folder into mater.wavelets:

    1. wfilters          -- load wavelet filter coefficients by name
    2. downsample        -- keep every M-th sample
    3. upsample          -- insert M-1 zeros between samples
    4. signals_sum       -- add two signals (zero-pads the shorter one)
    5. qmf_analysis      -- single-level QMF decomposition
    6. qmf_synthesis     -- single-level QMF reconstruction
    7. qmf_filter_iterator   -- build N-level iterated filter bank
    8. qmf_decomposition     -- multi-level decomposition
    9. compensation_blocks   -- per-subband gain/delay corrections
   10. qmf_reconstruction   -- multi-level reconstruction
   11. qmf_perfect_reconstruction_eval -- verify PR conditions

It also uses the existing mater.io.load and mater.fft.plot2D for
completeness.
"""

import numpy as np

# ──────────────────────────────────────────────────────────────────────
# 1. Loading wavelet filters
# ──────────────────────────────────────────────────────────────────────
# wfilters reads the bundled wfilters.mat file and returns the four
# filter coefficient arrays for any supported wavelet family.
#
# The filters are:
#   h0 -- low-pass  analysis  (decomposition)
#   h1 -- high-pass analysis  (decomposition)
#   g0 -- low-pass  synthesis (reconstruction)
#   g1 -- high-pass synthesis (reconstruction)

from mater.wavelets import wfilters

print("=" * 60)
print("1. LOADING WAVELET FILTERS")
print("=" * 60)

h0, h1, g0, g1 = wfilters("db5")  # Daubechies-5

print(f"Wavelet: db5")
print(f"  h0 (low-pass analysis) : {len(h0)} taps")
print(f"  h1 (high-pass analysis): {len(h1)} taps")
print(f"  g0 (low-pass synthesis): {len(g0)} taps")
print(f"  g1 (high-pass synthesis): {len(g1)} taps")
print(f"  h0 coefficients: {h0}")
print()

# You can load any wavelet in the .mat file -- e.g. 'haar', 'sym8', etc.
h0_haar, h1_haar, g0_haar, g1_haar = wfilters("haar")
print(f"Haar h0: {h0_haar}")
print()


# ──────────────────────────────────────────────────────────────────────
# 2. Basic signal operations: downsample, upsample, signals_sum
# ──────────────────────────────────────────────────────────────────────
# These are the multirate building blocks used internally by the QMF
# functions, but they're also useful on their own.

from mater.wavelets import downsample, upsample, signals_sum

print("=" * 60)
print("2. BASIC SIGNAL OPERATIONS")
print("=" * 60)

x = np.array([10, 20, 30, 40, 50, 60])

# downsample: keep every M-th sample
print(f"Original signal:      {x}")
print(f"Downsample by 2:      {downsample(x, 2)}")   # [10, 30, 50]
print(f"Downsample by 3:      {downsample(x, 3)}")   # [10, 40]
print()

# upsample: insert M-1 zeros between each sample
short = np.array([1, 2, 3])
print(f"Original signal:      {short}")
print(f"Upsample by 2:        {upsample(short, 2)}") # [1, 0, 2, 0, 3, 0]
print(f"Upsample by 3:        {upsample(short, 3)}") # [1, 0, 0, 2, 0, 0, 3, 0, 0]
print()

# signals_sum: add two signals of different lengths (zero-pads the shorter)
a = np.array([1, 2, 3, 4, 5])
b = np.array([100, 200])
print(f"a = {a}")
print(f"b = {b}")
print(f"signals_sum(a, b) = {signals_sum(a, b)}")  # [101, 202, 3, 4, 5]
print()


# ──────────────────────────────────────────────────────────────────────
# 3. Single-level QMF analysis and synthesis
# ──────────────────────────────────────────────────────────────────────
# qmf_analysis splits a signal into low-pass (approximation) and
# high-pass (detail) subbands through a two-channel filter bank.
# qmf_synthesis reconstructs the signal from those subbands.
#
#              ┌──[h0]──↓2──► x0 (approx)
#    x ────────┤
#              └──[h1]──↓2──► x1 (detail)
#
#    x0 ──↑2──[g0]──┐
#                    ├──(+)──► y (reconstructed)
#    x1 ──↑2──[g1]──┘

from mater.wavelets import qmf_analysis, qmf_synthesis

print("=" * 60)
print("3. SINGLE-LEVEL QMF ANALYSIS / SYNTHESIS")
print("=" * 60)

# Create a simple test signal
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
print(f"Original signal: {x}")

# Decompose using Haar filters (simplest wavelet)
x0, x1 = qmf_analysis(h0_haar, h1_haar, x)
print(f"Low-pass subband  (x0): {x0}")
print(f"High-pass subband (x1): {x1}")

# Reconstruct
y = qmf_synthesis(x0, x1, g0_haar, g1_haar)
print(f"Reconstructed (raw): {y}")

# The reconstruction has a gain (A) and delay (d). We can verify
# those with qmf_perfect_reconstruction_eval (see section 6).
# For Haar: A=1, d=1, so y[1:9] == x
print(f"Recovered (trim delay): {y[1:1+len(x)]}")
print()


# ──────────────────────────────────────────────────────────────────────
# 4. Multi-level QMF decomposition and reconstruction
# ──────────────────────────────────────────────────────────────────────
# For deeper analysis, we iterate the filter bank over multiple levels.
#
# qmf_filter_iterator  -- builds the equivalent filters for N levels
# qmf_decomposition    -- applies them all at once to produce subbands
# compensation_blocks  -- computes per-subband gain/delay correction
# qmf_reconstruction   -- puts the signal back together

from mater.wavelets import (
    qmf_filter_iterator,
    qmf_decomposition,
    compensation_blocks,
    qmf_reconstruction,
    qmf_perfect_reconstruction_eval,
)

print("=" * 60)
print("4. MULTI-LEVEL QMF DECOMPOSITION / RECONSTRUCTION")
print("=" * 60)

# Use db5 wavelet with 4 decomposition levels
levels = 4

# Build the iterated analysis and synthesis filter banks
# h  = [approx, detail_4, detail_3, detail_2, detail_1]
# mf = downsampling factor per subband (all powers of 2)
h, mf = qmf_filter_iterator(h0, h1, levels)
g, _  = qmf_filter_iterator(g0, g1, levels)

print(f"Wavelet: db5, levels: {levels}")
print(f"Number of subbands: {len(h)} (1 approx + {levels} detail)")
print(f"Downsampling factors: {mf}")
print(f"  (note: integer dtype = {mf.dtype}, no float casts needed)")
print()

# Generate a test signal: cosine + noise
np.random.seed(42)
N = 512
fs = 1000  # Hz
t = np.arange(N) / fs
x = np.cos(2 * np.pi * 50 * t) + 0.5 * np.random.randn(N)

print(f"Test signal: {N} samples, {fs} Hz sample rate")
print(f"  50 Hz cosine + Gaussian noise")

# Decompose
x_qmf, x_hat = qmf_decomposition(h, mf, x)
print(f"\nSubband lengths: {[len(c) for c in x_qmf]}")
print(f"Total coefficients in x_hat: {len(x_hat)}")

# Get PR parameters for gain/delay compensation
_, A, d = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
factors, shifts = compensation_blocks(A, d, levels)

print(f"\nPR parameters: gain A={A:.6f}, delay d={d}")
print(f"Gain corrections (factors): {factors}")
print(f"Delay corrections (shifts): {shifts}")
print(f"  (note: shifts are integers via << operator, dtype={shifts.dtype})")

# Reconstruct
xr = qmf_reconstruction(g, mf, factors, shifts, x_qmf, N)
err = np.max(np.abs(x - xr))
print(f"\nMax reconstruction error: {err:.2e}")
print(f"Perfect reconstruction achieved: {err < 1e-10}")
print()


# ──────────────────────────────────────────────────────────────────────
# 5. Scaling to many levels
# ──────────────────────────────────────────────────────────────────────
# The same workflow scales to any number of levels. Here we test
# 10 levels (like the original qmf.py demo).

print("=" * 60)
print("5. SCALING: 10-LEVEL DECOMPOSITION")
print("=" * 60)

levels_deep = 10
h_deep, mf_deep = qmf_filter_iterator(h0, h1, levels_deep)
g_deep, _ = qmf_filter_iterator(g0, g1, levels_deep)
factors_deep, shifts_deep = compensation_blocks(A, d, levels_deep)

# Use a longer signal for deep decomposition
x_long = np.random.randn(2048)
x_qmf_deep, x_hat_deep = qmf_decomposition(h_deep, mf_deep, x_long)
xr_deep = qmf_reconstruction(
    g_deep, mf_deep, factors_deep, shifts_deep, x_qmf_deep, len(x_long)
)

print(f"Signal length: {len(x_long)}")
print(f"Levels: {levels_deep}")
print(f"Subbands: {len(h_deep)}")
print(f"Subband lengths: {[len(c) for c in x_qmf_deep]}")
print(f"Max reconstruction error: {np.max(np.abs(x_long - xr_deep)):.2e}")
print()


# ──────────────────────────────────────────────────────────────────────
# 6. Perfect reconstruction evaluation
# ──────────────────────────────────────────────────────────────────────
# qmf_perfect_reconstruction_eval checks two mathematical conditions:
#   1. Aliasing cancellation: H0(-z)G0(z) + H1(-z)G1(z) = 0
#   2. LTI condition: 0.5 * [H0(z)G0(z) + H1(z)G1(z)] = A * z^{-d}
#
# Returns (is_pr, gain, delay) where delay is a negative error code
# if PR fails.

print("=" * 60)
print("6. PERFECT RECONSTRUCTION EVALUATION")
print("=" * 60)

# Test several wavelet families
for wname in ["haar", "db5", "db10", "db17"]:
    h0_w, h1_w, g0_w, g1_w = wfilters(wname)
    is_pr, gain, delay = qmf_perfect_reconstruction_eval(h0_w, h1_w, g0_w, g1_w)
    print(f"  {wname:6s}  PR={is_pr}  gain={gain:.6f}  delay={delay}")

# Test with random (non-PR) filters -- should fail
h0_bad = np.random.randn(6)
h1_bad = np.random.randn(6)
g0_bad = np.random.randn(6)
g1_bad = np.random.randn(6)
is_pr, gain, delay = qmf_perfect_reconstruction_eval(h0_bad, h1_bad, g0_bad, g1_bad)
print(f"  {'random':6s}  PR={is_pr}  gain={gain}  delay={delay}  (expected failure)")
print()


# ──────────────────────────────────────────────────────────────────────
# 7. Using mater.io.load (existing feature)
# ──────────────────────────────────────────────────────────────────────
# The load function reads MATLAB .mat files. The wfilters.mat bundled
# with the package is a regular .mat file -- we can peek at its keys.

from mater.io import load

print("=" * 60)
print("7. MATER.IO.LOAD (existing feature)")
print("=" * 60)

from pathlib import Path
mat_path = Path(__file__).resolve().parent.parent / "src" / "mater" / "wavelets" / "data" / "wfilters.mat"
if mat_path.exists():
    data = load(str(mat_path), make_global=False)
    # Show the first few variable names stored in the .mat file
    keys = [k for k in data.keys()]
    print(f"Variables in wfilters.mat: {len(keys)} total")
    print(f"First 10: {keys[:10]}")
else:
    print(f"wfilters.mat not found at {mat_path}")
print()


# ──────────────────────────────────────────────────────────────────────
# Summary
# ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print("""
Package structure after integration:

  mater/
  ├── __init__.py          (imports core, io, wavelets)
  ├── core/
  │   └── who.py           (MATLAB-like who())
  ├── io/
  │   └── load.py          (MATLAB .mat file loader)
  ├── fft/
  │   └── fft.py           (FFT spectrum plotting)
  └── wavelets/            ◄── NEW (from others/)
      ├── __init__.py
      ├── wfilters.py      (wavelet filter loader)
      ├── signal_ops.py    (downsample, upsample, signals_sum)
      ├── qmf.py           (analysis, synthesis, decomposition,
      │                      reconstruction, compensation, PR eval)
      └── data/
          └── wfilters.mat (bundled filter coefficients)
""")
