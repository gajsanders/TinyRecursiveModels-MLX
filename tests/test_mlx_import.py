import pytest


def test_mlx_import():
    """Test that mlx imports correctly and exposes mlx.__version__."""
    try:
        import mlx
    except ImportError:
        pytest.skip("mlx not available on this platform")
    
    # Assert that mlx.__version__ is available
    assert hasattr(mlx, '__version__'), "mlx should have __version__ attribute"
    assert mlx.__version__ is not None, "mlx.__version__ should not be None"
    assert isinstance(mlx.__version__, str), "mlx.__version__ should be a string"