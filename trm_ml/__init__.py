"""trm_ml package for TinyRecursiveModels-MLX.

This package provides the repository-local MLX shim and utilities used by
the test-suite and example scripts. Expose a stable __version__ used by
tests that check package metadata.
"""

__version__ = "0.1.0"

# Re-export device module
try:
    from trm_ml.device import get_device
except ImportError:
    # Fallback implementation
    def get_device():
        """Get the available device for computation.
        
        Returns:
            str: "mlx" if MLX is available, otherwise "cpu"
        """
        try:
            import mlx.core
            return "mlx"
        except ImportError:
            return "cpu"

__all__ = ["__version__", "get_device"]