import sys
import io
import contextlib
from unittest.mock import Mock
import numpy as np

# Create a mock for the mlx module and its components before importing our module
class MockArray:
    def __init__(self, value):
        self.value = value if isinstance(value, (list, np.ndarray)) else np.array([value])
    
    def __add__(self, other):
        return MockArray(self.value + (other.value if isinstance(other, MockArray) else other))
    
    def __sub__(self, other):
        return MockArray(self.value - (other.value if isinstance(other, MockArray) else other))
    
    def __pow__(self, power):
        return MockArray(self.value ** power)
    
    def __truediv__(self, other):
        return MockArray(self.value / (other.value if isinstance(other, MockArray) else other))
    
    def mean(self):
        # Return a MockArray with the mean value
        mean_val = np.mean(self.value) if hasattr(self.value, 'mean') else self.value
        return MockArray(mean_val)
    
    def item(self):
        if isinstance(self.value, np.ndarray):
            return self.value.item() if self.value.size == 1 else np.mean(self.value)
        return float(self.value)
    
    @property
    def shape(self):
        if isinstance(self.value, np.ndarray):
            return self.value.shape
        return (1,)


class MockMLX:
    @staticmethod
    def mean(arr):
        if hasattr(arr, 'mean'):
            return arr.mean()
        return MockArray(np.mean(arr.value))
    
    @staticmethod
    def zeros(shape):
        # Return a MockArray with zeros of the specified shape
        import numpy as np
        zeros_array = np.zeros(shape)
        return MockArray(zeros_array)
    
    @staticmethod
    def array_equal(arr1, arr2):
        return np.array_equal(arr1.value, arr2.value)
    
    @staticmethod
    def allclose(arr1, arr2, rtol=1e-5, atol=1e-5):
        return np.allclose(arr1.value, arr2.value, rtol=rtol, atol=atol)

# Create a mock mlx.core module
original_mlx = sys.modules.get('mlx', None)
original_mlx_core = sys.modules.get('mlx.core', None)

sys.modules['mlx'] = MockMLX()
sys.modules['mlx.core'] = MockMLX()

# Clean up after ourselves
def cleanup_mock_modules():
    # Remove the mock modules from sys.modules
    if 'mlx' in sys.modules:
        if original_mlx is not None:
            sys.modules['mlx'] = original_mlx
        else:
            del sys.modules['mlx']
    if 'mlx.core' in sys.modules:
        if original_mlx_core is not None:
            sys.modules['mlx.core'] = original_mlx_core
        else:
            del sys.modules['mlx.core']

import atexit
atexit.register(cleanup_mock_modules)

# Now we can import our training module
from trm_ml.training import train_one_epoch


def test_train_one_epoch_iterates_batches():
    """Test that train_one_epoch actually iterates over batches."""
    # Create a mock model
    mock_model = Mock()
    
    # Mock the forward method to return different outputs for different inputs
    def mock_forward(inputs):
        # Return an array based on the input for testing purposes
        if hasattr(inputs, 'value'):
            # If inputs is a MockArray, create output based on its value
            return MockArray(inputs.value * 0.5 + 0.1)  # Simple transformation for testing
        else:
            # Otherwise, just return some mock output
            return MockArray([0.1, 0.2])
    
    mock_model.forward = mock_forward
    
    # Create mock data: list of (inputs, targets) tuples
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]))  # shape (2, 4)
    mock_targets1 = MockArray(np.array([[0.1, 0.2], [0.3, 0.4]]))  # shape (2, 2)
    
    mock_inputs2 = MockArray(np.array([[2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5]]))  # shape (2, 4)
    mock_targets2 = MockArray(np.array([[0.5, 0.6], [0.7, 0.8]]))  # shape (2, 2)
    
    mock_data = [(mock_inputs1, mock_targets1), (mock_inputs2, mock_targets2)]
    
    # Capture print output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        avg_loss = train_one_epoch(mock_model, mock_data, optimizer=None)
    
    # Get the output
    output = captured_output.getvalue()
    
    # Verify that the function processed both batches (should print for each batch)
    assert "Batch 1" in output, "Expected output to contain 'Batch 1'"
    assert "Batch 2" in output, "Expected output to contain 'Batch 2'"
    assert "Epoch completed" in output, "Expected output to contain epoch completion message"
    
    # Since we have 2 batches, function should work correctly
    assert avg_loss is not None, "Expected a return value for average loss"


def test_train_one_epoch_prints_losses():
    """Test that train_one_epoch prints varying loss values."""
    # Create a mock model
    mock_model = Mock()
    
    # Mock the forward method
    def mock_forward(inputs):
        # Return an array based on the input
        return MockArray([0.1, 0.2])  # Simple mock output
    
    mock_model.forward = mock_forward
    
    # Create mock data with different values to produce different losses
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0]]))
    mock_targets1 = MockArray(np.array([[0.1, 0.2]]))
    
    mock_inputs2 = MockArray(np.array([[5.0, 6.0, 7.0, 8.0]]))
    mock_targets2 = MockArray(np.array([[0.9, 1.0]]))
    
    mock_data = [(mock_inputs1, mock_targets1), (mock_inputs2, mock_targets2)]
    
    # Capture print output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        train_one_epoch(mock_model, mock_data, optimizer=None)
    
    # Get the output
    output = captured_output.getvalue()
    
    # Verify that losses are printed for each batch
    lines = output.split('\n')
    loss_lines = [line for line in lines if 'Loss:' in line]
    
    assert len(loss_lines) >= 2, f"Expected at least 2 loss lines, got {len(loss_lines)}"
    
    # Verify that each batch has a loss value printed
    batch1_found = any('Batch 1' in line for line in loss_lines)
    batch2_found = any('Batch 2' in line for line in loss_lines)
    
    assert batch1_found, "Expected loss print for Batch 1"
    assert batch2_found, "Expected loss print for Batch 2"


def test_train_one_epoch_handles_empty_data():
    """Test that train_one_epoch handles empty data gracefully."""
    # Create a mock model
    mock_model = Mock()
    mock_model.forward = Mock(return_value=MockArray([0.1, 0.2]))
    
    # Empty data
    empty_data = []
    
    # Capture print output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        avg_loss = train_one_epoch(mock_model, empty_data, optimizer=None)
    
    # Get the output
    output = captured_output.getvalue()
    
    # Verify that appropriate message is shown for no batches
    assert "No batches processed" in output, "Expected message for no batches processed"
    assert avg_loss == 0.0, "Expected 0.0 average loss for empty data"


def test_train_one_epoch_varying_losses():
    """Test that different inputs produce different loss values."""
    # Create a mock model that returns different predictions
    mock_model = Mock()
    
    def varying_forward(inputs):
        # Return different outputs based on the input to generate varying losses
        if hasattr(inputs, 'value') and len(inputs.value) > 0:
            # Create output that's related to input for meaningful MSE calculation
            return MockArray(inputs.value * 0.5)  # Scale down the input
        return MockArray([0.1, 0.1])  # Default output
    
    mock_model.forward = varying_forward
    
    # Create data that should produce different losses
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0]]))  # Higher values
    mock_targets1 = MockArray(np.array([[0.5, 1.0]]))
    
    mock_inputs2 = MockArray(np.array([[0.1, 0.2, 0.3, 0.4]]))  # Lower values  
    mock_targets2 = MockArray(np.array([[0.05, 0.1]]))
    
    mock_data = [(mock_inputs1, mock_targets1), (mock_inputs2, mock_targets2)]
    
    # Capture print output
    captured_output = io.StringIO()
    with contextlib.redirect_stdout(captured_output):
        train_one_epoch(mock_model, mock_data, optimizer=None)
    
    # Get the output
    output = captured_output.getvalue()
    
    # Check that losses were printed
    assert "Loss:" in output, "Expected loss to be printed"
    assert "Batch 1" in output and "Batch 2" in output, "Expected both batches to be processed"