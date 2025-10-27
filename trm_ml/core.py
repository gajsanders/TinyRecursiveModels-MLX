"""Lightweight shim for MLX core used by tests.

This provides a minimal subset of the MLX API used by the unit tests
and the local `mlx` package. It intentionally uses NumPy under the hood so
tests can run without the external `mlx` runtime present.

Exports (small subset):
- array, zeros, mean, float32, asnumpy (helper)

If you later install the real MLX runtime, remove or adapt this shim.
"""
from __future__ import annotations

import numpy as _np
from typing import Any, Iterable, Optional, Tuple


float32 = _np.float32


def array(obj: Any, dtype: Optional[Any] = None) -> _np.ndarray:
    """Create an array using NumPy as a stand-in for MLX arrays.

    The tests expect an object with .shape and .dtype and that can be
    passed to model.forward. Using NumPy satisfies those expectations.
    """
    if dtype is None:
        dtype = float32
    return _np.array(obj, dtype=dtype)


def zeros(shape: Iterable[int] | Tuple[int, ...], dtype: Optional[Any] = None) -> _np.ndarray:
    if dtype is None:
        dtype = float32
    return _np.zeros(shape, dtype=dtype)


def mean(x: _np.ndarray, axis: Optional[int] = None):
    return _np.mean(x, axis=axis)


def asnumpy(x: Any) -> _np.ndarray:
    """Return a NumPy array for the given object (noop for NumPy)."""
    return _np.array(x)


# Provide a small random API (tests call mx.random.normal(...))
random = _np.random
