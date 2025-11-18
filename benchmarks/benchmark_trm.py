#!/usr/bin/env python3
"""
Benchmark script for Tiny Recursive Model (TRM)

This script instantiates the TRM model, runs a sample forward pass,
and measures elapsed time and RAM usage.
"""

import importlib.util
import os
import sys
import time

import numpy as np
import psutil

# Add the installed packages to path first to ensure proper MLX import
sys.path.insert(0, "/Users/enceladus/Library/Python/3.9/lib/python/site-packages")

# Now use importlib to import the TRM module directly from file

# Import the installed MLX library first
import trm_ml.core as mx

model_trm_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "trm_ml", "model_trm.py"
)
spec = importlib.util.spec_from_file_location("model_trm", model_trm_path)
model_trm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_trm)

# Get the TRM class from the imported module
TRM = model_trm.TRM


def benchmark_trm():
    """Benchmark TRM model performance."""
    print("Initializing TRM model...")

    # Define model parameters
    input_dim = 128
    latent_dim = 256
    output_dim = 64

    # Get initial memory usage
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB

    # Instantiate TRM model
    start_time = time.time()
    model = TRM(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim)
    init_time = time.time() - start_time

    print(f"Model initialized in {init_time:.4f} seconds")
    print(
        f"Model parameters: input_dim={input_dim}, latent_dim={latent_dim}, output_dim={output_dim}"
    )

    # Create sample input data
    batch_size = 32
    x = mx.array(np.random.rand(batch_size, input_dim).astype(np.float32))
    y = mx.array(
        np.random.rand(batch_size, output_dim).astype(np.float32)
    )  # For recursive reasoning

    # Measure memory after initialization but before forward pass
    memory_after_init = process.memory_info().rss / 1024 / 1024  # MB

    print("\nRunning forward pass benchmark...")
    # Run forward pass
    start_time = time.time()
    output = model.forward(x)
    forward_time = time.time() - start_time

    # Measure memory after forward pass
    memory_after_forward = process.memory_info().rss / 1024 / 1024  # MB

    print(f"Forward pass completed in {forward_time:.4f} seconds")
    print(f"Output shape: {output.shape}")

    # Run recursive reasoning benchmark
    print("\nRunning recursive reasoning benchmark...")
    num_steps = 10
    start_time = time.time()
    final_state = model.recursive_reasoning(x, y, steps=num_steps)
    recursive_time = time.time() - start_time

    # Measure memory after recursive reasoning
    memory_after_recursive = process.memory_info().rss / 1024 / 1024  # MB

    print(
        f"Recursive reasoning ({num_steps} steps) completed in {recursive_time:.4f} seconds"
    )
    print(f"Final state shape: {final_state.shape}")

    # Print results
    print("\n" + "=" * 50)
    print("BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Initialization time: {init_time:.4f} seconds")
    print(f"Forward pass time: {forward_time:.4f} seconds")
    print(f"Recursive reasoning time ({num_steps} steps): {recursive_time:.4f} seconds")
    print(f"Average time per recursive step: {recursive_time/num_steps:.4f} seconds")
    print("\nMemory Usage (MB):")
    print(f"Initial memory: {initial_memory:.2f}")
    print(
        f"After model init: {memory_after_init:.2f} (Δ: {memory_after_init - initial_memory:.2f})"
    )
    print(
        f"After forward: {memory_after_forward:.2f} (Δ: {memory_after_forward - memory_after_init:.2f})"
    )
    print(
        f"After recursive: {memory_after_recursive:.2f} (Δ: {memory_after_recursive - memory_after_forward:.2f})"
    )
    print(f"Total memory increase: {memory_after_recursive - initial_memory:.2f} MB")


if __name__ == "__main__":
    benchmark_trm()
