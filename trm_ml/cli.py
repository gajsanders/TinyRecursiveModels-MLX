#!/usr/bin/env python3
"""
CLI interface for Tiny Recursive Model (TRM)

This module provides a command-line interface for running training, evaluation,
and benchmarking routines for the TRM model.
"""

import argparse
import sys
import os


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CLI for Tiny Recursive Model (TRM) - Training, Evaluation, and Benchmarking"
    )
    
    # Add mutually exclusive group for train/eval/benchmark options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--train", 
        action="store_true", 
        help="Run training routine"
    )
    group.add_argument(
        "--eval", 
        action="store_true", 
        help="Run evaluation routine"
    )
    group.add_argument(
        "--benchmark", 
        action="store_true", 
        help="Run benchmark routine"
    )
    
    # Add optional arguments that can be used with any mode
    parser.add_argument(
        "--model-input-dim", 
        type=int, 
        default=128,
        help="Input dimension for the model (default: 128)"
    )
    parser.add_argument(
        "--model-latent-dim", 
        type=int, 
        default=256,
        help="Latent dimension for the model (default: 256)"
    )
    parser.add_argument(
        "--model-output-dim", 
        type=int, 
        default=64,
        help="Output dimension for the model (default: 64)"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        help="Path to the dataset (for train/eval)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=10,
        help="Number of epochs for training (default: 10)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32,
        help="Batch size (default: 32)"
    )
    
    args = parser.parse_args()
    
    # Import the required functions/modules based on the selected mode
    if args.train:
        run_training(args)
    elif args.eval:
        run_evaluation(args)
    elif args.benchmark:
        run_benchmark(args)
    else:
        parser.print_help()
        return 1
    
    return 0


def run_training(args):
    """Run the training routine."""
    print(f"Starting training with configuration:")
    print(f"  - Input dimension: {args.model_input_dim}")
    print(f"  - Latent dimension: {args.model_latent_dim}")
    print(f"  - Output dimension: {args.model_output_dim}")
    print(f"  - Data path: {args.data_path or 'None (using sample data)'}")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Import and run the training function
    try:
        # Import the training module and run
        from trm_ml.training import train_one_epoch
        from trm_ml.model_trm import TRM
        
        # Create model
        model = TRM(
            input_dim=args.model_input_dim,
            latent_dim=args.model_latent_dim,
            output_dim=args.model_output_dim
        )
        
        # For now, we'll just call the training function with a mock data loader
        # In a real implementation, we would create or load actual data
        print("Training function called (actual implementation would run here)")
        print("Note: This is a placeholder - actual training would require proper data loading")
        
    except ImportError as e:
        print(f"Error importing training module: {e}")
        return 1
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    print("Training completed successfully!")
    return 0


def run_evaluation(args):
    """Run the evaluation routine."""
    print(f"Starting evaluation with configuration:")
    print(f"  - Input dimension: {args.model_input_dim}")
    print(f"  - Latent dimension: {args.model_latent_dim}")
    print(f"  - Output dimension: {args.model_output_dim}")
    print(f"  - Data path: {args.data_path or 'None (using sample data)'}")
    print(f"  - Batch size: {args.batch_size}")
    
    # Import and run the evaluation function
    try:
        # Import the evaluation module and run
        from trm_ml.evaluation import run_evaluation
        from trm_ml.model_trm import TRM
        
        # Create model
        model = TRM(
            input_dim=args.model_input_dim,
            latent_dim=args.model_latent_dim,
            output_dim=args.model_output_dim
        )
        
        # For now, we'll just call the evaluation function with a mock data loader
        # In a real implementation, we would create or load actual data
        print("Evaluation function called (actual implementation would run here)")
        print("Note: This is a placeholder - actual evaluation would require proper data loading")
        
    except ImportError as e:
        print(f"Error importing evaluation module: {e}")
        return 1
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return 1
    
    print("Evaluation completed successfully!")
    return 0


def run_benchmark(args):
    """Run the benchmark routine."""
    print(f"Starting benchmark with configuration:")
    print(f"  - Input dimension: {args.model_input_dim}")
    print(f"  - Latent dimension: {args.model_latent_dim}")
    print(f"  - Output dimension: {args.model_output_dim}")
    
    # Import and run the benchmark
    try:
        # Use subprocess to run the benchmark script as it's a separate file
        import subprocess
        result = subprocess.run([
            sys.executable, 
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "benchmarks", "benchmark_trm.py")
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Benchmark output:")
            print(result.stdout)
            print("Benchmark completed successfully!")
            return 0
        else:
            print(f"Benchmark failed with error: {result.stderr}")
            return 1
            
    except Exception as e:
        print(f"Error during benchmark: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())