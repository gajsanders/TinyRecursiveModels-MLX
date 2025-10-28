import numpy as np
import trm_ml.core as mx
from trm_ml.model_trm import TRM


def test_initialize_state_shape():
    """Test that initialize_state() returns the correct shape."""
    # Create a TRM model
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    # Test various batch sizes
    for batch_size in [1, 5, 10, 16]:
        state = model.initialize_state(batch_size)
        
        # Verify the shape is (batch_size, latent_dim)
        assert state.shape == (batch_size, model.latent_dim), \
            f"Expected shape ({batch_size}, {model.latent_dim}), got {state.shape}"


def test_initialize_state_values():
    """Test that initialize_state() returns all zeros."""
    model = TRM(input_dim=3, latent_dim=5, output_dim=2)
    
    # Test with different batch sizes
    for batch_size in [1, 3, 7]:
        state = model.initialize_state(batch_size)
        
        # Convert to numpy to check values easily
        state_np = np.array(state)
        
        # Verify all values are zeros
        expected_zeros = np.zeros((batch_size, model.latent_dim))
        assert np.array_equal(state_np, expected_zeros), \
            f"Expected all zeros, got {state_np}"


def test_initialize_state_dtype():
    """Test that initialize_state() returns the correct dtype."""
    model = TRM(input_dim=4, latent_dim=6, output_dim=3)
    
    batch_size = 5
    state = model.initialize_state(batch_size)
    
    # Verify the dtype is float32 (default for mx.zeros)
    assert state.dtype == mx.float32, \
        f"Expected dtype mx.float32, got {state.dtype}"


def test_initialize_state_latent_dim_consistency():
    """Test that initialize_state() respects the model's latent_dim."""
    # Test with different latent dimensions
    for latent_dim in [1, 4, 16, 32]:
        model = TRM(input_dim=5, latent_dim=latent_dim, output_dim=3)
        state = model.initialize_state(batch_size=2)
        
        assert state.shape[1] == latent_dim, \
            f"Expected latent_dim {latent_dim}, got {state.shape[1]}"


def test_initialize_state_batch_size_parameter():
    """Test that initialize_state() respects the batch_size parameter."""
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    # Test various batch sizes
    batch_sizes = [1, 2, 5, 10, 20]
    for batch_size in batch_sizes:
        state = model.initialize_state(batch_size)
        assert state.shape[0] == batch_size, \
            f"Expected batch_size {batch_size}, got {state.shape[0]}"


def test_update_state_shape():
    """Test that update_state() returns the correct shape."""
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    # Initialize state and create input tensors
    batch_size = 5
    state = model.initialize_state(batch_size)
    x = mx.random.normal((batch_size, model.input_dim))
    y = mx.random.normal((batch_size, model.output_dim))
    
    # Call update_state
    updated_state = model.update_state(state, x, y)
    
    # Verify the shape matches the original state
    assert updated_state.shape == state.shape, \
        f"Expected shape {state.shape}, got {updated_state.shape}"
    
    # Test with different batch sizes
    for test_batch_size in [1, 3, 7, 10]:
        state = model.initialize_state(test_batch_size)
        x = mx.random.normal((test_batch_size, model.input_dim))
        y = mx.random.normal((test_batch_size, model.output_dim))
        
        updated_state = model.update_state(state, x, y)
        assert updated_state.shape == (test_batch_size, model.latent_dim), \
            f"Expected shape ({test_batch_size}, {model.latent_dim}), got {updated_state.shape}"


def test_update_state_changes_when_x_changes():
    """Test that update_state output actually changes when x changes."""
    model = TRM(input_dim=4, latent_dim=8, output_dim=2)
    
    batch_size = 5
    state = model.initialize_state(batch_size)
    y = mx.random.normal((batch_size, model.output_dim))
    
    # Create two different x inputs
    x1 = mx.random.normal((batch_size, model.input_dim))
    x2 = mx.random.normal((batch_size, model.input_dim))
    
    # Call update_state with both inputs
    updated_state_1 = model.update_state(state, x1, y)
    updated_state_2 = model.update_state(state, x2, y)
    
    # Convert to numpy for easier comparison
    state_1_np = np.array(updated_state_1)
    state_2_np = np.array(updated_state_2)
    
    # Verify that the two outputs are different when x is different
    # Using a small tolerance for floating point comparison
    are_equal = np.allclose(state_1_np, state_2_np, rtol=1e-5, atol=1e-5)
    assert not are_equal, \
        "update_state output should be different when x changes, but got identical outputs"
    
    # Additional test: Verify that when x is the same, output is the same
    updated_state_3 = model.update_state(state, x1, y)
    state_3_np = np.array(updated_state_3)
    
    # outputs from same x input should be the same
    assert np.allclose(state_1_np, state_3_np, rtol=1e-5, atol=1e-5), \
        "update_state output should be the same when x is the same"