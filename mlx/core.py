"""Re-export of `trm_ml.core` for backwards compatibility.

This module imports everything from `trm_ml.core` so code that does
`import mlx.core as mx` keeps working unchanged.
"""
from importlib import import_module

# Import the real implementation
_core = import_module("trm_ml.core")

# Re-export public names
from trm_ml.core import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", [])
