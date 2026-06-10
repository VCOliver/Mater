# mater

**Package to make Python commands closer to MATLAB's.**

`mater` is a Python library that brings familiar MATLAB-style workflow patterns to Python: loading `.mat` files directly into the namespace, inspecting workspace variables, plotting FFT spectra, and working with wavelet filter banks via QMF analysis and synthesis.

---

## Installation

**From PyPI:**

```bash
pip install mater
```

**From source (using [uv](https://docs.astral.sh/uv/)):**

```bash
git clone https://github.com/VCOliver/mater.git
cd mater
uv sync
```

**Requirements:** Python >= 3.12

---

## Module Overview

| Subpackage        | Description                                                                      |
|-------------------|----------------------------------------------------------------------------------|
| `mater.core`      | Workspace inspection (`who`) — lists user-defined variables like MATLAB's `who`  |
| `mater.io`        | File I/O (`load`) — loads MATLAB `.mat` files directly into the namespace        |
| `mater.fft`       | Frequency-domain plotting (`plot2D`) — FFT magnitude spectrum                    |
| `mater.wavelets`  | Wavelet filter banks — filter loading, multirate ops, QMF analysis/synthesis     |

---

## Usage

### `mater.io.load` — Load `.mat` files

Loads variables from a MATLAB `.mat` file. By default, all variables are injected into the caller's global namespace (equivalent to MATLAB's `load`).

```python
from mater.io import load

# Load all variables into the global namespace
load('data.mat')
print(signal)  # variable from the .mat file is now available

# Load specific variables, returned as values
x, t = load('data.mat', 'x', 't')

# Load without injecting into globals; get a dict instead
data = load('data.mat', make_global=False)
print(data.keys())

# Load a specific variable as a dict (slices=False)
data = load('data.mat', 'x', slices=False)
print(data['x'])
```

---

### `mater.core.who` — List workspace variables

Displays user-defined variables in the current namespace, filtering out built-ins, imports, and functions — mirroring MATLAB's `who`.

```python
from mater import *

x = 42
y = [1, 2, 3]
name = "hello"

who()
# Your variables are:
#
# name    x    y
```

Variables present before `mater` was imported are excluded automatically.

---

### `mater.fft.plot2D` — FFT spectrum plot

Computes and plots the two-sided FFT magnitude spectrum of a signal.

```python
import numpy as np
from mater.fft import plot2D

fs = 1000  # sampling frequency in Hz
t = np.linspace(0, 1, fs, endpoint=False)
x = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

plot2D(x, fs=fs, title="Signal Spectrum")
```

**Parameters:**
- `arr` — input signal (NumPy array or list)
- `fs` — sampling frequency in Hz
- `title` — plot title (default: `"Frequency Spectrum"`)

---

### `mater.wavelets` — Wavelet filter banks

The wavelets subpackage provides the building blocks for discrete wavelet transforms using QMF (Quadrature Mirror Filter) filter banks, mirroring MATLAB's Wavelet Toolbox.

#### `wfilters` — Load wavelet filter coefficients

```python
from mater.wavelets import wfilters

h0, h1, g0, g1 = wfilters('db5')
# h0: low-pass decomposition filter
# h1: high-pass decomposition filter
# g0: low-pass reconstruction filter
# g1: high-pass reconstruction filter
```

Supports standard wavelet families (e.g. `'db5'`, `'sym8'`, `'haar'`, `'coif3'`), matching MATLAB's `wfilters('db5')`.

---

#### `downsample` / `upsample` — Multirate signal operations

```python
import numpy as np
from mater.wavelets import downsample, upsample

x = np.array([1, 2, 3, 4, 5, 6])

downsample(x, 2)   # array([1, 3, 5])   — keep every 2nd sample
upsample(x, 3)     # array([1, 0, 0, 2, 0, 0, 3, 0, 0, ...])  — insert M-1 zeros between samples
```

---

#### `signals_sum` — Length-safe signal addition

Adds two signals of potentially different lengths by zero-padding the shorter one.

```python
from mater.wavelets import signals_sum
import numpy as np

signals_sum(np.array([1, 2, 3]), np.array([10, 20]))
# array([11, 22,  3])
```

---

#### `qmf_analysis` / `qmf_synthesis` — Single-level QMF filter bank

```python
import numpy as np
from mater.wavelets import wfilters, qmf_analysis, qmf_synthesis

h0, h1, g0, g1 = wfilters('haar')
x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# Decompose into low-pass (approximation) and high-pass (detail) subbands
x0, x1 = qmf_analysis(h0, h1, x)

# Reconstruct from subbands
y = qmf_synthesis(x0, x1, g0, g1)
```

---

#### `qmf_filter_iterator` — Build iterated N-level filter bank

Computes the equivalent filters for an N-level wavelet decomposition tree, so the entire decomposition can be applied in a single pass.

```python
from mater.wavelets import wfilters, qmf_filter_iterator

h0, h1, g0, g1 = wfilters('db5')

h, multirate_factors = qmf_filter_iterator(h0, h1, levels=3)
# h: list of 4 equivalent filters [approx, detail_3, detail_2, detail_1]
# multirate_factors: array([8, 8, 4, 2])
```

---

#### `qmf_decomposition` — Multi-level decomposition

```python
import numpy as np
from mater.wavelets import wfilters, qmf_filter_iterator, qmf_decomposition

h0, h1, g0, g1 = wfilters('db5')
h, mf = qmf_filter_iterator(h0, h1, levels=3)

x = np.random.randn(256)
x_qmf, x_hat = qmf_decomposition(h, mf, x)
# x_qmf: list of per-subband coefficient arrays
# x_hat: all coefficients concatenated into one vector
```

---

#### `compensation_blocks` — Gain and delay corrections

Computes the per-subband gain and delay compensation needed for perfect reconstruction after multi-level decomposition.

```python
from mater.wavelets import wfilters, qmf_perfect_reconstruction_eval, compensation_blocks

h0, h1, g0, g1 = wfilters('db5')
is_pr, A, d = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)

factors, shifts = compensation_blocks(A=A, d=d, levels=3)
```

---

#### `qmf_reconstruction` — Multi-level reconstruction

```python
import numpy as np
from mater.wavelets import (
    wfilters, qmf_filter_iterator, qmf_decomposition,
    qmf_perfect_reconstruction_eval, compensation_blocks,
    qmf_reconstruction,
)

h0, h1, g0, g1 = wfilters('db5')
levels = 3

h, mf = qmf_filter_iterator(h0, h1, levels)
g, _  = qmf_filter_iterator(g0, g1, levels)
_, A, d = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)
fct, sh = compensation_blocks(A, d, levels)

x = np.random.randn(256)
x_qmf, _ = qmf_decomposition(h, mf, x)
xr = qmf_reconstruction(g, mf, fct, sh, x_qmf, len(x))

np.allclose(x, xr)  # True
```

---

#### `qmf_perfect_reconstruction_eval` — Verify perfect reconstruction

Checks whether a set of four filters satisfies the aliasing-cancellation and distortion-free conditions for perfect reconstruction.

```python
from mater.wavelets import wfilters, qmf_perfect_reconstruction_eval

h0, h1, g0, g1 = wfilters('db5')
is_pr, gain, delay = qmf_perfect_reconstruction_eval(h0, h1, g0, g1)

print(is_pr)   # True
print(gain)    # overall gain constant A
print(delay)   # sample delay d introduced by the filter bank
```

Returns `(is_pr, gain, delay)`. When perfect reconstruction fails, `delay` contains an error code: `-2` (aliasing not cancelled), `-4` (zero gain), or `-5` (LTI condition failed).

---

## License

MIT License — Copyright (c) 2025 Victor Cruz. See [LICENSE](LICENSE) for details.
