import numpy as np
import pytest


@pytest.fixture
def sample_input_data():
    """Pytest fixture that loads sample_input.npy and checks its shape."""
    arr = np.load("tests/data/sample_input.npy")
    assert arr.shape == (2, 4), f"Expected shape (2, 4), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected dtype float32, got {arr.dtype}"
    return arr


@pytest.fixture
def sample_target_data():
    """Pytest fixture that loads sample_target.npy and checks its shape."""
    arr = np.load("tests/data/sample_target.npy")
    assert arr.shape == (2, 1), f"Expected shape (2, 1), got {arr.shape}"
    assert arr.dtype == np.float32, f"Expected dtype float32, got {arr.dtype}"
    return arr


# Simple tests to verify the fixtures work
def test_sample_input_fixture(sample_input_data):
    assert sample_input_data.shape == (2, 4)
    assert sample_input_data.dtype == np.float32


def test_sample_target_fixture(sample_target_data):
    assert sample_target_data.shape == (2, 1)
    assert sample_target_data.dtype == np.float32