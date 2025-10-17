import sys
import io
import logging
from contextlib import redirect_stderr
from mlx.utils.logging_utils import log_msg, debug, info, warning, error, critical


def test_log_msg_default_level():
    """Test that log_msg uses INFO level by default."""
    # Capture stderr to check log output
    captured_output = io.StringIO()
    
    # Temporarily redirect stderr to capture logging output
    with redirect_stderr(captured_output):
        log_msg("Test message")
        
    output = captured_output.getvalue()
    
    # Check that the output contains INFO level
    assert "INFO" in output, f"Expected INFO level in output: {output}"
    assert "Test message" in output, f"Expected message in output: {output}"


def test_log_msg_custom_levels():
    """Test that log_msg works with different log levels."""
    for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
        captured_output = io.StringIO()
        
        with redirect_stderr(captured_output):
            log_msg(f"Test {level} message", level=level)
            
        output = captured_output.getvalue()
        
        assert level in output, f"Expected {level} level in output: {output}"
        assert f"Test {level} message" in output, f"Expected message in output: {output}"


def test_log_msg_invalid_level_defaults_to_info():
    """Test that invalid log levels default to INFO."""
    captured_output = io.StringIO()
    
    with redirect_stderr(captured_output):
        log_msg("Test default message", level="INVALID_LEVEL")
        
    output = captured_output.getvalue()
    
    # Should default to INFO
    assert "INFO" in output, f"Expected INFO level (default) in output: {output}"
    assert "Test default message" in output, f"Expected message in output: {output}"


def test_convenience_functions():
    """Test the convenience logging functions."""
    level_functions = [
        (debug, "DEBUG"),
        (info, "INFO"), 
        (warning, "WARNING"),
        (error, "ERROR"),
        (critical, "CRITICAL")
    ]
    
    for func, expected_level in level_functions:
        captured_output = io.StringIO()
        
        with redirect_stderr(captured_output):
            func(f"Test {expected_level} message")
            
        output = captured_output.getvalue()
        
        assert expected_level in output, f"Expected {expected_level} level in output: {output}"
        assert f"Test {expected_level} message" in output, f"Expected message in output: {output}"


def test_debug_message_capture():
    """Test that debug messages are properly captured."""
    # Set logging level to DEBUG to ensure debug messages are shown
    logging.getLogger().setLevel(logging.DEBUG)
    
    captured_output = io.StringIO()
    
    with redirect_stderr(captured_output):
        debug("Debug message for testing")
        
    output = captured_output.getvalue()
    
    assert "DEBUG" in output, f"Expected DEBUG level in output: {output}"
    assert "Debug message for testing" in output, f"Expected debug message in output: {output}"
    
    # Reset logging level
    logging.getLogger().setLevel(logging.INFO)


def test_log_format():
    """Test that log messages have the expected format."""
    captured_output = io.StringIO()
    
    with redirect_stderr(captured_output):
        log_msg("Format test message", level="WARNING")
        
    output = captured_output.getvalue()
    
    # Check that the output contains timestamp, level, and message
    assert "WARNING" in output, "Expected WARNING level in output"
    assert "Format test message" in output, "Expected message in output"
    # Note: Testing for timestamp format would be complex due to time variations


def test_log_msg_case_insensitive_levels():
    """Test that log levels are treated case-insensitively."""
    for level in ["info", "Info", "INFO", "iNfO"]:
        captured_output = io.StringIO()
        
        with redirect_stderr(captured_output):
            log_msg("Case test message", level=level)
            
        output = captured_output.getvalue()
        
        assert "INFO" in output, f"Expected INFO level in output for case {level}: {output}"
        assert "Case test message" in output, f"Expected message in output for case {level}: {output}"


def test_log_msg_with_special_characters():
    """Test that log_msg handles special characters properly."""
    special_msg = "Test message with special chars: !@#$%^&*()"
    
    captured_output = io.StringIO()
    
    with redirect_stderr(captured_output):
        log_msg(special_msg, level="INFO")
        
    output = captured_output.getvalue()
    
    assert "INFO" in output, f"Expected INFO level in output: {output}"
    assert special_msg in output, f"Expected special message in output: {output}"


def test_multiple_log_messages():
    """Test that multiple log messages work correctly."""
    captured_output = io.StringIO()
    
    with redirect_stderr(captured_output):
        log_msg("First message", level="INFO")
        log_msg("Second message", level="WARNING")
        debug("Debug message")
        
    output = captured_output.getvalue()
    
    # Check that all messages appear in the output
    assert "INFO" in output and "First message" in output
    assert "WARNING" in output and "Second message" in output
    # Note: Debug might not appear unless logging level is set low enough