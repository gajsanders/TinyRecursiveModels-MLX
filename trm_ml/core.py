"""MLX Core wrapper for Tiny Recursive Models.

This module provides a unified interface to MLX functionality,
with fallback to NumPy for testing environments where MLX is not available.
"""

from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, Tuple

_mlx_core = None
if "mlx.core" in sys.modules and sys.modules["mlx.core"] is not None:
    # tests sometimes inject a Mock object into sys.modules['mlx.core']
    _mlx_core = sys.modules["mlx.core"]
elif (
    "mlx" in sys.modules
    and sys.modules["mlx"] is not None
    and hasattr(sys.modules["mlx"], "core")
):
    _mlx_core = getattr(sys.modules["mlx"], "core")
else:
    try:
        # Try normal import
        import mlx.core as _imported_mlx_core  # type: ignore

        _mlx_core = _imported_mlx_core
    except Exception:
        _mlx_core = None

if _mlx_core is not None:
    HAS_MLX = True
    # Import numpy for fallback functions
    import numpy as _np

    # Use the available MLX-like implementation (could be a Mock)
    array = getattr(_mlx_core, "array", lambda obj, dtype=None: obj)
    zeros = getattr(
        _mlx_core,
        "zeros",
        lambda shape, dtype=None: _np.zeros(
            shape, dtype=float32 if dtype is None else dtype
        ),
    )
    mean = getattr(
        _mlx_core,
        "mean",
        lambda x, axis=None, keepdims=False: x.mean() if hasattr(x, "mean") else x,
    )
    float32 = getattr(_mlx_core, "float32", _np.float32)
    asnumpy = getattr(
        _mlx_core, "asnumpy", getattr(_mlx_core, "to_numpy", lambda x: _np.array(x))
    )

    # Custom RandomModule that wraps MLX's random module
    class RandomModule:
        def __init__(self, mlx_random):
            self._mlx_random = mlx_random

        def normal(self, shape):
            # Ensure shape is properly handled
            if isinstance(shape, int):
                shape = (shape,)
            try:
                # Try to call the MLX random function
                return self._mlx_random.normal(shape)
            except Exception:
                # Fall back to NumPy if MLX fails
                import numpy as _np

                return _np.random.normal(0, 1, shape)

    random = RandomModule(getattr(_mlx_core, "random", None))

    def array_equal(a, b):
        # Prefer the MLX implementation if available
        if _mlx_core is not None and hasattr(_mlx_core, "array_equal"):
            try:
                return _mlx_core.array_equal(a, b)
            except Exception:
                pass
        # Fall back to NumPy - even if _mlx_core is None
        try:
            import numpy as _np

            # Convert to numpy arrays if they have .value attributes
            a_np = a.value if hasattr(a, "value") else a
            b_np = b.value if hasattr(b, "value") else b
            return _np.array_equal(_np.array(a_np), _np.array(b_np))
        except Exception:
            return False

    def allclose(a, b, rtol=1e-05, atol=1e-08):
        # Prefer the MLX implementation if available
        if _mlx_core is not None and hasattr(_mlx_core, "allclose"):
            try:
                return _mlx_core.allclose(a, b, rtol=rtol, atol=atol)
            except Exception:
                pass
        # Fall back to NumPy - even if _mlx_core is None
        try:
            import numpy as _np

            # Convert to numpy arrays if they have .value attributes
            a_np = a.value if hasattr(a, "value") else a
            b_np = b.value if hasattr(b, "value") else b
            return _np.allclose(_np.array(a_np), _np.array(b_np), rtol=rtol, atol=atol)
        except Exception:
            # Last resort: compare element-wise
            try:
                import numpy as _np

                a_flat = _np.array(a.value if hasattr(a, "value") else a).flatten()
                b_flat = _np.array(b.value if hasattr(b, "value") else b).flatten()
                return _np.allclose(a_flat, b_flat, rtol=rtol, atol=atol)
            except Exception:
                return False

    if hasattr(_mlx_core, "__version__"):
        __version__ = _mlx_core.__version__
else:
    # Fall back to a NumPy-based implementation
    import numpy as _np

    HAS_MLX = False
    float32 = _np.float32

    def _flatten_numeric(obj):
        """Flatten nested iterables into a 1-D list of numbers."""
        if obj is None:
            return []
        if hasattr(obj, "value"):
            return _flatten_numeric(obj.value)
        if isinstance(obj, (list, tuple)):
            out = []
            for v in obj:
                out.extend(_flatten_numeric(v))
            return out
        try:
            # scalar or numpy scalar
            return [float(obj)]
        except Exception:
            return []

    def array(obj: Any, dtype: Optional[Any] = None) -> Any:
        """Create a NumPy array, flattening heterogeneous sequences when necessary."""
        if dtype is None:
            dtype = float32
        # If obj is a list of arrays with different shapes, flatten to 1-D numeric list
        if isinstance(obj, (list, tuple)):
            try:
                return _np.array(obj, dtype=dtype)
            except Exception:
                flat = _flatten_numeric(obj)
                return _np.array(flat, dtype=dtype)
        # For other objects, attempt direct conversion
        try:
            return _np.array(obj, dtype=dtype)
        except Exception:
            flat = _flatten_numeric(obj)
            return _np.array(flat, dtype=dtype)

    def zeros(
        shape: Iterable[int] | Tuple[int, ...], dtype: Optional[Any] = None
    ) -> Any:
        if dtype is None:
            dtype = float32
        return _np.zeros(shape, dtype=dtype)

    def mean(x: Any, axis: Optional[int] = None, keepdims: bool = False):
        # If object has mean method that accepts no axis kw, call it and coerce to numpy float
        try:
            m = x.mean
        except Exception:
            return _np.mean(_np.array(x))
        else:
            try:
                return m(axis=axis, keepdims=keepdims)
            except TypeError:
                # fallback: use numpy
                return _np.mean(_np.array(x))

    def asnumpy(x: Any) -> Any:
        return _np.array(x, dtype=float32)

    def array_equal(a, b):
        # Use numpy for comparison
        try:
            import numpy as _np

            # Convert to numpy arrays if they have .value attributes
            a_np = a.value if hasattr(a, "value") else a
            b_np = b.value if hasattr(b, "value") else b
            return _np.array_equal(_np.array(a_np), _np.array(b_np))
        except Exception:
            return False

    def allclose(a, b, rtol=1e-05, atol=1e-08):
        # Use numpy for comparison
        try:
            import numpy as _np

            # Convert to numpy arrays if they have .value attributes
            a_np = a.value if hasattr(a, "value") else a
            b_np = b.value if hasattr(b, "value") else b
            return _np.allclose(_np.array(a_np), _np.array(b_np), rtol=rtol, atol=atol)
        except Exception:
            # Last resort: compare element-wise
            try:
                import numpy as _np

                a_flat = _np.array(a.value if hasattr(a, "value") else a).flatten()
                b_flat = _np.array(b.value if hasattr(b, "value") else b).flatten()
                return _np.allclose(a_flat, b_flat, rtol=rtol, atol=atol)
            except Exception:
                return False

    # Provide a small random API (tests call mx.random.normal(...))
    class RandomModule:
        def normal(self, shape):
            # Ensure shape is properly handled
            if isinstance(shape, int):
                shape = (shape,)
            return _np.random.normal(0, 1, shape)

    random = RandomModule()

    # Provide version for compatibility
    __version__ = "0.1.0"
