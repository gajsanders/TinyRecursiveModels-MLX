import importlib.util
import sys


def get_device():
    """
    Get the available device for computation.
    
    Returns:
        str: "mlx" if MLX is available, otherwise "cpu"
    """
    try:
        # First check if mlx.core is explicitly disabled in sys.modules (for testing)
        if 'mlx.core' in sys.modules and sys.modules['mlx.core'] is None:
            return 'cpu'
            
        # If a real system 'mlx' runtime is installed we prefer that
        if importlib.util.find_spec('mlx.core') is not None:
            return 'mlx'
        if importlib.util.find_spec('mlx') is not None:
            return 'mlx'
    except Exception:
        # If find_spec fails, fall back to import check
        try:
            import mlx.core
            return 'mlx'
        except ImportError:
            pass
    return 'cpu'