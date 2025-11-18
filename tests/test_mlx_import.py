import pytest


def test_mlx_import():
    """Test that trm_ml imports correctly and exposes trm_ml.__version__."""
    try:
        import trm_ml
    except ImportError:
        pytest.skip("trm_ml not available on this platform")
    
    # Assert that trm_ml.__version__ is available
    assert hasattr(trm_ml, '__version__'), "trm_ml should have __version__ attribute"
    assert trm_ml.__version__ is not None, "trm_ml.__version__ should not be None"
    assert isinstance(trm_ml.__version__, str), "trm_ml.__version__ should be a string"