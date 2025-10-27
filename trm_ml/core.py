"""MLX Core wrapper for Tiny Recursive Models.

This module provides a unified interface to MLX functionality,
with fallback to NumPy for testing environments where MLX is not available.
"""
from __future__ import annotations

import sys
from typing import Any, Iterable, Optional, Tuple

# Try to import the real MLX library, fallback to NumPy-based shim
try:
    import mlx.core as _mlx_core
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    # Use NumPy as fallback for testing
    import numpy as _mlx_core
    import numpy as _np


if HAS_MLX:
    # Use real MLX functionality
    array = _mlx_core.array
    zeros = _mlx_core.zeros
    mean = _mlx_core.mean
    float32 = _mlx_core.float32
    
    # Only add asnumpy if it exists in MLX
    if hasattr(_mlx_core, 'asnumpy'):
        asnumpy = _mlx_core.asnumpy
    else:
        # Create a fallback for MLX if asnumpy doesn't exist
        def asnumpy(x):
            return _mlx_core.to_numpy(x) if hasattr(_mlx_core, 'to_numpy') else x
    
    random = _mlx_core.random
    
    # Add compatibility functions needed by tests
    def array_equal(a, b):
        if hasattr(_mlx_core, 'array_equal'):
            return _mlx_core.array_equal(a, b)
        else:
            # Convert to numpy for comparison if MLX lacks array_equal
            a_np = _mlx_core.to_numpy(a) if hasattr(_mlx_core, 'to_numpy') else a
            b_np = _mlx_core.to_numpy(b) if hasattr(_mlx_core, 'to_numpy') else b
            import numpy as _np
            return _np.array_equal(a_np, b_np)
    
    def allclose(a, b, rtol=1e-05, atol=1e-08):
        if hasattr(_mlx_core, 'allclose'):
            return _mlx_core.allclose(a, b, rtol=rtol, atol=atol)
        else:
            # Convert to numpy for comparison if MLX lacks allclose
            a_np = _mlx_core.to_numpy(a) if hasattr(_mlx_core, 'to_numpy') else a
            b_np = _mlx_core.to_numpy(b) if hasattr(_mlx_core, 'to_numpy') else b
            import numpy as _np
            return _np.allclose(a_np, b_np, rtol=rtol, atol=atol)
    
    # Re-export other MLX functionality as needed
    if hasattr(_mlx_core, '__version__'):
        __version__ = _mlx_core.__version__
else:
    # Fallback NumPy-based implementation for testing
    float32 = _np.float32
    
    def array(obj: Any, dtype: Optional[Any] = None) -> Any:
        """Create an array, using MLX if available or NumPy as fallback."""
        if dtype is None:
            dtype = float32
        return _np.array(obj, dtype=dtype)
    
    def zeros(shape: Iterable[int] | Tuple[int, ...], dtype: Optional[Any] = None) -> Any:
        if dtype is None:
            dtype = float32
        return _np.zeros(shape, dtype=dtype)
    
    def mean(x: Any, axis: Optional[int] = None, keepdims: bool = False):
        return _np.mean(x, axis=axis, keepdims=keepdims)
    
    def asnumpy(x: Any) -> Any:
        """Return a NumPy array."""
        return _np.array(x)
    
    def array_equal(a, b):
        return _np.array_equal(a, b)
    
    def allclose(a, b, rtol=1e-05, atol=1e-08):
        return _np.allclose(a, b, rtol=rtol, atol=atol)
    
    # Provide a random API (tests call mx.random.normal(...))
    class RandomModule:
        def normal(self, shape):
            return _np.random.normal(0, 1, shape)
    
    random = RandomModule()