from mater.core import *
from mater.io import *
from mater import wavelets

# Snapshot the REPL's globals right after our names are injected.
# This lets who() distinguish "was already there" from "user-created".
from mater.core.who import _capture_initial_globals as _cig
_cig()
del _cig