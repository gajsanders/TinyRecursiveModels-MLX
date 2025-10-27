#!/usr/bin/env python3
"""
Test script for benchmark functionality.

This test verifies that the benchmark script runs without error 
and produces expected outputs for timing and RAM usage.
"""

import subprocess
import sys
import os
import pytest


def test_benchmark_script_runs_without_error():
    """Test that the benchmark script runs without raising errors."""
    # Path to the benchmark script
    benchmark_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "benchmarks", 
        "benchmark_trm.py"
    )
    
    # Ensure the script exists
    assert os.path.exists(benchmark_script), f"Benchmark script not found: {benchmark_script}"
    
    # Run the benchmark script as a subprocess
    result = subprocess.run(
        [sys.executable, benchmark_script],
        capture_output=True,
        text=True,
        timeout=60  # Set timeout to avoid hanging
    )
    
    # Check that the script ran successfully (return code 0)
    assert result.returncode == 0, f"Benchmark script failed with return code {result.returncode}. Error: {result.stderr}"
    
    # Check that we got some output
    assert result.stdout, "Benchmark script produced no output"
    
    # Verify that the output contains expected benchmark elements
    output = result.stdout.lower()
    
    # Check for timing-related keywords
    assert "time" in output, "Output should contain timing information"
    assert "seconds" in output or "s" in output, "Output should contain time measurements"
    
    # Check for memory/RAM-related keywords
    assert "memory" in output or "ram" in output, "Output should contain memory usage information"
    assert "mb" in output, "Output should contain memory measurements in MB"
    
    # Check for specific benchmark sections
    assert "benchmark results" in output, "Output should contain benchmark results section"
    assert "initialization time" in output or "init time" in output, "Output should contain initialization timing"
    assert "forward pass" in output, "Output should contain forward pass timing"
    
    print("Benchmark script test passed!")
    print("Output preview:")
    print("-" * 50)
    print(result.stdout[-1000:])  # Print last 1000 characters of output


def test_benchmark_script_outputs_timing_and_memory():
    """Test that the benchmark script outputs proper timing and memory usage."""
    # Path to the benchmark script
    benchmark_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "benchmarks", 
        "benchmark_trm.py"
    )
    
    # Run the benchmark script
    result = subprocess.run(
        [sys.executable, benchmark_script],
        capture_output=True,
        text=True,
        timeout=60
    )
    
    assert result.returncode == 0, f"Benchmark script failed: {result.stderr}"
    
    output = result.stdout
    
    # Verify specific output patterns
    import re
    
    # Check for timing measurements (should be in seconds)
    time_pattern = r"\d+\.?\d*\s*seconds"
    time_matches = re.findall(time_pattern, output)
    assert len(time_matches) > 0, f"Expected timing information in output, found: {time_matches}"
    
    # Check for memory usage in MB - look for various formats
    # Format 1: "MB" after a number (e.g., "Total memory increase: 1.23 MB")
    memory_pattern1 = r"\d+\.?\d+\s*mb"
    # Format 2: Numbers in memory section that represent MB values
    memory_pattern2 = r"Memory Usage \(MB\):"
    
    memory_matches1 = re.findall(memory_pattern1, output.lower())
    memory_header_match = re.search(memory_pattern2, output)
    
    # We should have either direct MB references or the memory section header
    assert (len(memory_matches1) > 0 or memory_header_match is not None), \
        f"Expected memory information in output, found MB patterns: {memory_matches1}, header: {memory_header_match is not None}"
    
    # Also check for the specific memory usage lines
    memory_usage_lines = [
        "Initial memory:" in output,
        "After model init:" in output, 
        "After forward:" in output,
        "After recursive:" in output,
        "Total memory increase:" in output
    ]
    
    # Should have most of these memory tracking lines
    assert sum(memory_usage_lines) >= 4, f"Expected multiple memory usage tracking lines, found: {memory_usage_lines}"
    
    print(f"Found {len(time_matches)} timing measurements and {len(memory_matches1)} direct MB measurements")
    print("Timing values:", time_matches)
    print("Direct MB values:", memory_matches1)
    print("Memory header found:", memory_header_match is not None)
    print("Memory usage lines found:", sum(memory_usage_lines))


if __name__ == "__main__":
    # Run the tests if this script is executed directly
    test_benchmark_script_runs_without_error()
    test_benchmark_script_outputs_timing_and_memory()
    print("All benchmark tests passed!")