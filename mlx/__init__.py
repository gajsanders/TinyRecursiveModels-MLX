"""Compatibility shim for the historical `mlx` package.

This thin wrapper re-exports selected modules from `trm_ml` so existing
imports in tests and legacy code (e.g. `import mlx.core as mx`) continue to work
after renaming the main package to `trm_ml`.
"""
__all__ = ["core", "utils", "__version__"]

__version__ = "0.1.0"
# mlx package for TinyRecursiveModels-MLX project
# This package contains custom modules for the TRM project
# It should not be confused with the MLX framework package