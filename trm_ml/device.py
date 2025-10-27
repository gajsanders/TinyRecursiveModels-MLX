def get_device():
    """
    Get the available device for computation.
    
    Returns:
        str: "mlx" if MLX is available, otherwise "cpu"
    """
    try:
        # Try to import MLX to check if it's available
        import trm_ml.core
        # If import succeeds, return "mlx"
        return "mlx"
    except ImportError:
        # If import fails, return "cpu"
        return "cpu"