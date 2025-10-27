#!/usr/bin/env python3
"""
Test script for full integration pipeline.

This test verifies that the full pipeline (TRM + data loader + training + 
evaluation + logging) runs without errors and produces expected outputs.
"""

import subprocess
import sys
import os
import pytest
import numpy as np


def test_full_pipeline_runs_without_error():
    """Test that the full pipeline runs without raising errors."""
    # Path to the wire_up script
    wire_up_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "trm_ml", 
        "wire_up.py"
    )
    
    # Ensure the script exists
    assert os.path.exists(wire_up_script), f"Wire up script not found: {wire_up_script}"
    
    # Run the wire_up script with minimal parameters to test quickly
    result = subprocess.run([
        sys.executable, 
        wire_up_script,
        "--epochs", "1",  # Just 1 epoch for testing
        "--batch-size", "4",  # Small batch size for quick test
        "--input-dim", "16",  # Small dimensions for quick test
        "--latent-dim", "32",
        "--output-dim", "8"
    ], capture_output=True, text=True, timeout=120)
    
    # Check that the script ran successfully (return code 0)
    assert result.returncode == 0, f"Full pipeline failed with return code {result.returncode}. Error: {result.stderr}"
    
    # Check that we got expected training outputs
    output = result.stdout
    assert "batch" in output.lower(), "Expected batch training output not found"
    assert "epoch completed" in output.lower(), "Expected epoch completion message not found"
    assert "pipeline results:" in output.lower(), "Expected pipeline results header not found"
    assert "model config:" in output.lower(), "Expected model config in results not found"
    assert "training results:" in output.lower(), "Expected training results not found"
    assert "evaluation result:" in output.lower(), "Expected evaluation result not found"
    
    print("Full pipeline runs without error test passed!")
    print("Output:")
    print("-" * 50)
    print(output[-800:])  # Print last 800 chars of output


def test_full_pipeline_outputs_expected_results():
    """Test that the full pipeline outputs expected result structure."""
    # Import the wire_up module using subprocess to avoid import conflicts
    import subprocess
    import json
    
    # Run the wire_up script and capture the results programmatically
    # We'll modify wire_up to also return structured results when called in test mode
    wire_up_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "trm_ml", 
        "wire_up.py"
    )
    
    # For now, let's run a simple version and check if it works
    result = subprocess.run([
        sys.executable,
        wire_up_script,
        "--epochs", "1",
        "--batch-size", "4",
        "--input-dim", "16",
        "--latent-dim", "32",
        "--output-dim", "8"
    ], capture_output=True, text=True, timeout=60)
    
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    
    # The actual results are printed to stdout, so let's parse them
    output = result.stdout
    
    # Check that expected elements are in the output
    assert "Pipeline Results:" in output
    assert "Model Config:" in output
    assert "Training Results:" in output
    assert "Evaluation Result:" in output
    
    # Verify specific configuration values appear in output
    assert "input_dim': 16" in output or "'input_dim': 16" in output
    assert "latent_dim': 32" in output or "'latent_dim': 32" in output
    assert "output_dim': 8" in output or "'output_dim': 8" in output
    
    print(f"Pipeline results structure test passed!")
    print(f"  Found expected result structure in output")


def test_full_pipeline_with_logging():
    """Test that the full pipeline produces proper output during execution."""
    # Run the pipeline and check that it provides progress updates
    wire_up_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "trm_ml", 
        "wire_up.py"
    )
    
    result = subprocess.run([
        sys.executable, 
        wire_up_script,
        "--epochs", "1",
        "--batch-size", "4",
        "--input-dim", "8",
        "--latent-dim", "16", 
        "--output-dim", "4"
    ], capture_output=True, text=True, timeout=60)
    
    assert result.returncode == 0, f"Pipeline failed: {result.stderr}"
    
    # Check that we get training progress output
    output = result.stdout.lower()
    
    # Verify that training batches are processed (should see batch updates)
    assert "batch" in output, "Expected batch processing output"
    assert "loss:" in output or "batch" in output, "Expected loss or batch updates during training"
    assert "epoch completed" in output, "Expected epoch completion"
    
    # Verify final results are output
    assert "pipeline results:" in output, "Expected final pipeline results"
    
    print("Full pipeline progress output test passed!")


def test_pipeline_with_different_configurations():
    """Test the pipeline with different configurations to ensure robustness."""
    configurations = [
        {"input_dim": 4, "latent_dim": 8, "output_dim": 2, "epochs": 1, "batch_size": 2},
        {"input_dim": 32, "latent_dim": 64, "output_dim": 16, "epochs": 2, "batch_size": 8},
    ]
    
    wire_up_script = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "trm_ml", 
        "wire_up.py"
    )
    
    for i, config in enumerate(configurations):
        print(f"Testing configuration {i+1}: {config}")
        
        # Run the pipeline with specific configuration
        cmd = [sys.executable, wire_up_script]
        for key, value in config.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, f"Pipeline failed for config {i+1}: {result.stderr}"
        
        # Check that expected results are in the output
        output = result.stdout
        assert "pipeline results:" in output.lower()
        assert "training results:" in output.lower()
        assert "evaluation result:" in output.lower()
        
        # Verify specific configuration values appear in output
        config_str = str(config['input_dim'])
        assert config_str in output, f"Config value {config_str} not found in output"
        
        print(f"  Configuration {i+1} passed!")
    
    print("All pipeline configuration tests passed!")


if __name__ == "__main__":
    # Run all tests when executed directly
    test_full_pipeline_runs_without_error()
    test_full_pipeline_outputs_expected_results()
    test_full_pipeline_with_logging()
    test_pipeline_with_different_configurations()
    print("All full integration tests passed!")