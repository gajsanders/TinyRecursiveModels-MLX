import sys
import numpy as np
from unittest.mock import Mock

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
        mean_val = np.mean(self.value.flatten()) if hasattr(self.value, 'flatten') else np.mean(self.value)
        return MockArray(mean_val)
    
    def item(self):
        if isinstance(self.value, np.ndarray):
            return float(self.value.flatten()[0]) if self.value.size >= 1 else 0.0
        return float(self.value)
    
    @property
    def shape(self):
        if isinstance(self.value, np.ndarray):
            return self.value.shape
        return (1,)
    
    def tolist(self):
        return self.value.tolist() if hasattr(self.value, 'tolist') else [self.value]
    
    def flatten(self):
        return MockArray(self.value.flatten() if hasattr(self.value, 'flatten') else np.array([self.value]))


class MockMLX:
    @staticmethod
    def mean(arr):
        if hasattr(arr, 'mean'):
            return arr.mean()
        return MockArray(np.mean(arr.value))
    
    @staticmethod
    def zeros(shape):
        # Return a MockArray with zeros of the specified shape
        zeros_array = np.zeros(shape)
        return MockArray(zeros_array)
    
    @staticmethod
    def array_equal(arr1, arr2):
        return np.array_equal(arr1.value, arr2.value)
    
    @staticmethod
    def allclose(arr1, arr2, rtol=1e-5, atol=1e-5):
        return np.allclose(arr1.value, arr2.value, rtol=rtol, atol=atol)
    
    @staticmethod
    def array(data):
        # Convert data to a MockArray
        if isinstance(data, MockArray):
            return data
        return MockArray(np.array(data) if not isinstance(data, np.ndarray) else data)


# Create mock mlx.core module
sys.modules['mlx'] = MockMLX()
sys.modules['mlx.core'] = MockMLX()

# Now we can import our evaluation module
from mlx.evaluation import run_evaluation


def test_run_evaluation_returns_scalar():
    """Test that run_evaluation returns a scalar mean value."""
    # Create a mock model
    mock_model = Mock()
    
    # Mock the forward method to return different outputs for testing
    def mock_forward(inputs):
        # Return an array based on the input
        if hasattr(inputs, 'value'):
            # If inputs is a MockArray, return a transformation
            input_val = inputs.value
            # Return a 2D array similar to model outputs (batch_size, output_dim)
            if input_val.ndim == 2:
                batch_size, input_dim = input_val.shape
                output_dim = 2  # assume output dim is 2
                output = np.ones((batch_size, output_dim)) * 0.5
                return MockArray(output)
        # Default return
        return MockArray([[0.1, 0.2], [0.3, 0.4]])  # Shape (2, 2)
    
    mock_model.forward = mock_forward
    
    # Create mock data: list of input tensors
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]]))  # shape (2, 4)
    mock_inputs2 = MockArray(np.array([[2.0, 3.0, 4.0, 5.0], [1.5, 2.5, 3.5, 4.5]]))  # shape (2, 4)
    
    mock_data = [mock_inputs1, mock_inputs2]
    
    # Run evaluation
    result = run_evaluation(mock_model, mock_data)
    
    # Verify that result is a scalar (float or int)
    assert isinstance(result, (int, float)), f"Expected scalar result, got {type(result)}"
    
    # The result should be a meaningful mean value
    assert not (isinstance(result, bool) and result == True), "Result should not be boolean True"
    assert not (isinstance(result, bool) and result == False), "Result should not be boolean False"


def test_run_evaluation_with_tuple_data():
    """Test that run_evaluation handles (inputs, targets) tuple format."""
    # Create a mock model
    mock_model = Mock()
    
    def mock_forward(inputs):
        # Return a fixed output for consistent testing
        return MockArray([[0.1, 0.2], [0.3, 0.4]])  # Shape (2, 2)
    
    mock_model.forward = mock_forward
    
    # Create mock data as (inputs, targets) tuples
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0]]))  # shape (1, 4)
    mock_targets1 = MockArray(np.array([[0.1, 0.2]]))  # shape (1, 2)
    
    mock_inputs2 = MockArray(np.array([[2.0, 3.0, 4.0, 5.0]]))  # shape (1, 4)
    mock_targets2 = MockArray(np.array([[0.5, 0.6]]))  # shape (1, 2)
    
    mock_data = [(mock_inputs1, mock_targets1), (mock_inputs2, mock_targets2)]
    
    # Run evaluation
    result = run_evaluation(mock_model, mock_data)
    
    # Verify that result is a scalar
    assert isinstance(result, (int, float)), f"Expected scalar result, got {type(result)}"
    
    # The result should be the mean of all outputs: 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4 -> mean of 0.25
    # Each batch has outputs [[0.1, 0.2], [0.3, 0.4]] which gets flattened to [0.1, 0.2, 0.3, 0.4]
    # Two batches means we have two sets of [0.1, 0.2, 0.3, 0.4] -> [0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4]
    # Mean should be around 0.25


def test_run_evaluation_single_batch():
    """Test that run_evaluation works with a single batch."""
    # Create a mock model
    mock_model = Mock()
    
    def mock_forward(inputs):
        return MockArray([[0.5, 0.7]])  # Single batch, shape (1, 2)
    
    mock_model.forward = mock_forward
    
    # Single batch of data
    mock_inputs = MockArray(np.array([[1.0, 2.0, 3.0, 4.0]]))  # shape (1, 4)
    mock_data = [mock_inputs]
    
    # Run evaluation
    result = run_evaluation(mock_model, mock_data)
    
    # Verify that result is a scalar
    assert isinstance(result, (int, float)), f"Expected scalar result, got {type(result)}"
    
    # Should be mean of [0.5, 0.7] which is 0.6


def test_run_evaluation_empty_data():
    """Test that run_evaluation handles empty data gracefully."""
    # Create a mock model
    mock_model = Mock()
    mock_model.forward = Mock(return_value=MockArray([[0.1, 0.2]]))
    
    # Empty data
    empty_data = []
    
    # Run evaluation
    result = run_evaluation(mock_model, empty_data)
    
    # Should return 0.0 for empty data
    assert result == 0.0, f"Expected 0.0 for empty data, got {result}"


def test_run_evaluation_output_shape():
    """Test that the output is a scalar regardless of input/output shapes."""
    # Create a mock model
    mock_model = Mock()
    
    def mock_forward(inputs):
        # Return various shaped outputs to test
        if hasattr(inputs, 'value') and inputs.value.shape[0] == 3:
            return MockArray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])  # Shape (3, 3)
        else:
            return MockArray([[0.1, 0.2], [0.3, 0.4]])  # Shape (2, 2)
    
    mock_model.forward = mock_forward
    
    # Create data with different batch sizes and shapes
    mock_inputs1 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]))  # Shape (3, 4)
    mock_inputs2 = MockArray(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))  # Shape (2, 4)
    
    mock_data = [mock_inputs1, mock_inputs2]
    
    # Run evaluation
    result = run_evaluation(mock_model, mock_data)
    
    # Verify that result is a scalar
    assert isinstance(result, (int, float)), f"Expected scalar result, got {type(result)}"
    
    # Result should be a meaningful mean of all outputs
    assert not np.isnan(result), "Result should not be NaN"
    assert not np.isinf(result), "Result should not be infinite"


def test_run_evaluation_return_type():
    """Test that run_evaluation specifically returns a scalar value, not an array."""
    # Create a mock model
    mock_model = Mock()
    
    def mock_forward(inputs):
        return MockArray([[0.1, 0.2], [0.3, 0.4]])  # Shape (2, 2)
    
    mock_model.forward = mock_forward
    
    # Create mock data
    mock_inputs = MockArray(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]))  # Shape (2, 4)
    mock_data = [mock_inputs]
    
    # Run evaluation
    result = run_evaluation(mock_model, mock_data)
    
    # Verify it's a scalar, not an array or MockArray
    assert isinstance(result, (int, float)), f"Expected scalar (int/float), got {type(result)}"
    assert not hasattr(result, 'shape'), "Result should not have a shape attribute (should be scalar)"
    assert not hasattr(result, '__len__') or result == 0, "Result should not be a sequence"