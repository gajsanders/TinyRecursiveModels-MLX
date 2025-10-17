import pytest
from mlx.model_trm import TRM


def test_trm_init_valid_dimensions():
    """Test that TRM class can be instantiated with valid dimensions."""
    # Test with valid positive integer dimensions
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    # Verify that the attributes are set correctly
    assert model.input_dim == 4
    assert model.latent_dim == 8
    assert model.output_dim == 2


def test_trm_init_invalid_input_dim_type():
    """Test that TRM raises TypeError for invalid input_dim type."""
    with pytest.raises(TypeError, match="input_dim must be an integer"):
        TRM(input_dim="4", latent_dim=8, output_dim=2)
    
    with pytest.raises(TypeError, match="input_dim must be an integer"):
        TRM(input_dim=4.5, latent_dim=8, output_dim=2)
    
    with pytest.raises(TypeError, match="input_dim must be an integer"):
        TRM(input_dim=None, latent_dim=8, output_dim=2)


def test_trm_init_invalid_latent_dim_type():
    """Test that TRM raises TypeError for invalid latent_dim type."""
    with pytest.raises(TypeError, match="latent_dim must be an integer"):
        TRM(input_dim=4, latent_dim="8", output_dim=2)
    
    with pytest.raises(TypeError, match="latent_dim must be an integer"):
        TRM(input_dim=4, latent_dim=8.5, output_dim=2)
    
    with pytest.raises(TypeError, match="latent_dim must be an integer"):
        TRM(input_dim=4, latent_dim=None, output_dim=2)


def test_trm_init_invalid_output_dim_type():
    """Test that TRM raises TypeError for invalid output_dim type."""
    with pytest.raises(TypeError, match="output_dim must be an integer"):
        TRM(input_dim=4, latent_dim=8, output_dim="2")
    
    with pytest.raises(TypeError, match="output_dim must be an integer"):
        TRM(input_dim=4, latent_dim=8, output_dim=2.5)
    
    with pytest.raises(TypeError, match="output_dim must be an integer"):
        TRM(input_dim=4, latent_dim=8, output_dim=None)


def test_trm_init_invalid_dimensions_value():
    """Test that TRM raises ValueError for non-positive dimensions."""
    # Test input_dim <= 0
    with pytest.raises(ValueError, match="input_dim must be positive"):
        TRM(input_dim=0, latent_dim=8, output_dim=2)
    
    with pytest.raises(ValueError, match="input_dim must be positive"):
        TRM(input_dim=-1, latent_dim=8, output_dim=2)
    
    # Test latent_dim <= 0
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        TRM(input_dim=4, latent_dim=0, output_dim=2)
    
    with pytest.raises(ValueError, match="latent_dim must be positive"):
        TRM(input_dim=4, latent_dim=-1, output_dim=2)
    
    # Test output_dim <= 0
    with pytest.raises(ValueError, match="output_dim must be positive"):
        TRM(input_dim=4, latent_dim=8, output_dim=0)
    
    with pytest.raises(ValueError, match="output_dim must be positive"):
        TRM(input_dim=4, latent_dim=8, output_dim=-1)