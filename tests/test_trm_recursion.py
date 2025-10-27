import numpy as np
import mlx.core as mx
from trm_ml.model_trm import TRM


def test_recursive_reasoning_steps():
    """Test that recursive_reasoning performs the correct number of updates."""
    # Create a TRM model
    model = TRM(input_dim=4, latent_dim=6, output_dim=2)
    
    # Create sample input and target tensors
    batch_size = 3
    x = mx.random.normal((batch_size, model.input_dim))
    y = mx.random.normal((batch_size, model.output_dim))
    
    # Test different numbers of steps
    for steps in [0, 1, 5, 10]:
        final_state = model.recursive_reasoning(x, y, steps)
        
        # Verify the shape is correct (batch_size, latent_dim)
        assert final_state.shape == (batch_size, model.latent_dim), \
            f"Expected shape ({batch_size}, {model.latent_dim}), got {final_state.shape}"
        
        # For 0 steps, the final state should be the initialized state (all zeros)
        if steps == 0:
            expected_state = model.initialize_state(batch_size)
            assert mx.array_equal(final_state, expected_state), \
                "For 0 steps, final state should equal initialized state"


def test_recursive_reasoning_matches_manual_updates():
    """Test that recursive_reasoning final state equals the result of manual update_state calls."""
    # Create a TRM model
    model = TRM(input_dim=3, latent_dim=5, output_dim=2)
    
    # Create sample input and target tensors
    batch_size = 2
    x = mx.random.normal((batch_size, model.input_dim))
    y = mx.random.normal((batch_size, model.output_dim))
    
    # Test with different numbers of steps
    for steps in [1, 3, 7]:
        # Get result from recursive_reasoning
        final_state_from_method = model.recursive_reasoning(x, y, steps)
        
        # Manually perform the same sequence of updates
        state = model.initialize_state(batch_size)
        for _ in range(steps):
            state = model.update_state(state, x, y)
        
        # The results should be identical
        assert mx.allclose(final_state_from_method, state, rtol=1e-5, atol=1e-5), \
            f"For {steps} steps, recursive_reasoning result should match manual updates"


def test_recursive_reasoning_state_evolution():
    """Test that the state actually evolves during recursion steps."""
    # Create a TRM model
    model = TRM(input_dim=4, latent_dim=3, output_dim=2)
    
    # Create sample input and target tensors
    batch_size = 2
    x = mx.random.normal((batch_size, model.input_dim))
    y = mx.random.normal((batch_size, model.output_dim))
    
    # Get state after 1 step and after 2 steps
    state_after_1 = model.recursive_reasoning(x, y, 1)
    state_after_2 = model.recursive_reasoning(x, y, 2)
    
    # These should be different since we're applying another update
    assert not mx.allclose(state_after_1, state_after_2, rtol=1e-5, atol=1e-5), \
        "State after 1 step should be different from state after 2 steps"


def test_recursive_reasoning_deterministic():
    """Test that recursive_reasoning produces the same result with the same inputs."""
    # Create a TRM model
    model = TRM(input_dim=3, latent_dim=4, output_dim=2)
    
    # Create sample input and target tensors
    batch_size = 2
    x = mx.random.normal((batch_size, model.input_dim))
    y = mx.random.normal((batch_size, model.output_dim))
    steps = 5
    
    # Run recursive reasoning multiple times with the same inputs
    result1 = model.recursive_reasoning(x, y, steps)
    result2 = model.recursive_reasoning(x, y, steps)
    
    # Results should be identical
    assert mx.allclose(result1, result2, rtol=1e-5, atol=1e-5), \
        "Recursive reasoning should be deterministic for the same inputs"


def test_recursive_reasoning_batch_consistency():
    """Test recursive reasoning with various batch sizes."""
    # Create a TRM model
    model = TRM(input_dim=5, latent_dim=8, output_dim=3)
    
    # Test with different batch sizes
    for batch_size in [1, 4, 8]:
        x = mx.random.normal((batch_size, model.input_dim))
        y = mx.random.normal((batch_size, model.output_dim))
        steps = 3
        
        final_state = model.recursive_reasoning(x, y, steps)
        
        # Verify the shape is correct
        assert final_state.shape == (batch_size, model.latent_dim), \
            f"Expected shape ({batch_size}, {model.latent_dim}), got {final_state.shape}"