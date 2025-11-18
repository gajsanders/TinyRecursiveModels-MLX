"""trm_ml package for TinyRecursiveModels-MLX.

This package provides the repository-local MLX shim and utilities used by
the test-suite and example scripts. Expose a stable __version__ used by
tests that check package metadata.
"""

__version__ = "0.1.0"

# Explicitly import submodules to make them available
from . import core
from . import device
from . import model_trm
from . import training
from . import evaluation
from . import data_utils
from . import cli
from . import wire_up
from .utils import logging_utils


# Re-export device module function
def get_device():
    """Get the available device for computation.

    Returns:
        str: "mlx" if MLX is available, otherwise "cpu"
    """
    try:
        return device.get_device()
    except ImportError:
        # Fallback implementation
        return "cpu"


__all__ = [
    "__version__",
    "get_device",
    "core",
    "device",
    "model_trm",
    "training",
    "evaluation",
    "data_utils",
    "cli",
    "wire_up",
    "logging_utils",
]
