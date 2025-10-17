import platform
import pytest


def test_platform_is_arm():
    """Test that the platform processor is 'arm' (Apple Silicon), skip otherwise."""
    processor = platform.processor()
    if processor != "arm":
        pytest.skip(f"Test skipped: platform processor is '{processor}', expected 'arm' for Apple Silicon")
    assert processor == "arm"