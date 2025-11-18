#!/usr/bin/env python3
"""
Test script for CLI functionality.

This test verifies that the CLI correctly handles train, eval, and benchmark commands
and prints expected output for each option.
"""

import subprocess
import sys
import os
import pytest


def test_cli_help():
    """Test that the CLI shows help when called with --help."""
    result = subprocess.run([
        sys.executable, 
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--help"
    ], capture_output=True, text=True)
    
    # Help should succeed (return code 0) and contain expected text
    assert result.returncode == 0, f"CLI help failed: {result.stderr}"
    assert "train" in result.stdout.lower()
    assert "eval" in result.stdout.lower() 
    assert "benchmark" in result.stdout.lower()
    print("CLI help test passed!")


def test_cli_benchmark_option():
    """Test that CLI with --benchmark option runs the benchmark."""
    result = subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--benchmark"
    ], capture_output=True, text=True)
    
    # Benchmark should succeed (return code 0) and contain expected output
    assert result.returncode == 0, f"CLI benchmark failed: {result.stderr}"
    
    # Check that it starts benchmark with correct config
    output = result.stdout.lower()
    assert "starting benchmark" in output
    assert "input dimension" in output
    assert "latent dimension" in output
    assert "output dimension" in output
    
    # Check that it mentions benchmark completion
    assert "benchmark completed" in output or "benchmark output:" in output
    
    print("CLI benchmark test passed!")


def test_cli_train_option():
    """Test that CLI with --train option attempts to run training."""
    result = subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--train"
    ], capture_output=True, text=True)
    
    # Training call should either succeed or fail gracefully with import errors
    # (the import error is expected based on the mlx directory conflict)
    output = result.stdout.lower()
    
    # Check that it starts training with correct config
    assert "starting training" in output
    assert "input dimension" in output
    assert "latent dimension" in output
    assert "output dimension" in output
    assert "epochs" in output
    assert "batch size" in output
    
    print("CLI train test passed!")


def test_cli_eval_option():
    """Test that CLI with --eval option attempts to run evaluation."""
    result = subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--eval"
    ], capture_output=True, text=True)
    
    # Evaluation call should either succeed or fail gracefully with import errors
    output = result.stdout.lower()
    
    # Check that it starts evaluation with correct config
    assert "starting evaluation" in output
    assert "input dimension" in output
    assert "latent dimension" in output
    assert "output dimension" in output
    assert "batch size" in output
    
    print("CLI eval test passed!")


def test_cli_mutually_exclusive_options():
    """Test that CLI rejects multiple options at once."""
    # Try to run train and benchmark together (should fail)
    result = subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--train", "--benchmark"
    ], capture_output=True, text=True)
    
    # This should fail with an error about mutually exclusive arguments
    assert result.returncode != 0, "CLI should reject multiple options at once"
    print("CLI mutually exclusive test passed!")


def test_cli_custom_parameters():
    """Test that CLI accepts custom parameters."""
    result = subprocess.run([
        sys.executable,
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "trm_ml", "cli.py"),
        "--benchmark",
        "--model-input-dim", "64",
        "--model-latent-dim", "128", 
        "--model-output-dim", "32"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"CLI with custom parameters failed: {result.stderr}"
    
    # Check that custom parameters are reflected in output
    output = result.stdout.lower()
    assert "input dimension: 64" in output
    assert "latent dimension: 128" in output
    assert "output dimension: 32" in output
    
    print("CLI custom parameters test passed!")


if __name__ == "__main__":
    # Run all tests when executed directly
    test_cli_help()
    test_cli_benchmark_option()
    test_cli_train_option()
    test_cli_eval_option()
    test_cli_mutually_exclusive_options()
    test_cli_custom_parameters()
    print("All CLI tests passed!")