import numpy as np
import mlx.core as mx
from trm_ml.model_trm import TRM


def test_trm_forward_shape():
    """Test that forward() returns array of correct shape."""
    # Create a TRM model
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    # Create sample input data
    x = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])  # Shape: (2, 4)
    
    # Call forward method
    output = model.forward(x)
    
    # Verify output shape
    assert output.shape == (2, 2), f"Expected shape (2, 2), got {output.shape}"
    
    # Test with different batch size
    x_single = mx.array([[1.0, 2.0, 3.0, 4.0]])  # Shape: (1, 4)
    output_single = model.forward(x_single)
    assert output_single.shape == (1, 2), f"Expected shape (1, 2), got {output_single.shape}"
    
    # Test with larger batch size
    x_large = mx.array(np.random.rand(5, 4))  # Shape: (5, 4)
    output_large = model.forward(x_large)
    assert output_large.shape == (5, 2), f"Expected shape (5, 2), got {output_large.shape}"


def test_trm_forward_dtype():
    """Test that forward() returns array of correct dtype."""
    model = TRM(input_dim=4, latent_dim=8, output_dim=3)
    
    # Create sample input data
    x = mx.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    
    # Call forward method
    output = model.forward(x)
    
    # Verify output dtype (should be float32 by default for mx.zeros)
    assert output.dtype == mx.float32, f"Expected dtype mx.float32, got {output.dtype}"


def test_trm_forward_values():
    """Test that forward() returns zeros as expected."""
    model = TRM(input_dim=3, latent_dim=5, output_dim=2)
    
    # Create sample input data
    x = mx.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape: (3, 3)
    
    # Call forward method
    output = model.forward(x)
    
    # Verify output contains only zeros
    output_np = np.array(output)
    expected_zeros = np.zeros((3, 2))
    assert np.array_equal(output_np, expected_zeros), f"Expected all zeros, got {output_np}"


def test_trm_forward_batch_size_determination():
    """Test that forward() correctly determines batch size from input."""
    model = TRM(input_dim=2, latent_dim=4, output_dim=1)
    
    # Different input sizes to test batch size determination
    test_cases = [
        (1, 2),   # 1 sample
        (3, 2),   # 3 samples
        (10, 2),  # 10 samples
    ]
    
    for batch_size, input_dim in test_cases:
        x = mx.array(np.random.rand(batch_size, input_dim))
        output = model.forward(x)
        
        assert output.shape[0] == batch_size, f"Expected batch size {batch_size}, got {output.shape[0]}"
        assert output.shape[1] == model.output_dim, f"Expected output dim {model.output_dim}, got {output.shape[1]}"