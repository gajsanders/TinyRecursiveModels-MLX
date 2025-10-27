"""Compatibility re-export for `trm_ml.core`.

This module makes `import mlx.core as mx` work by forwarding symbols from
the `trm_ml.core` implementation.
"""
from importlib import import_module

_core = import_module("trm_ml.core")

# Re-export public names from the real implementation
from trm_ml.core import *  # noqa: F401,F403

__all__ = getattr(_core, "__all__", [])
