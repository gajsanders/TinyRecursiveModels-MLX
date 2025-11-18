# TinyRecursiveModels-MLX

**Mac MLX Compatible Version**

A port of the recursive reasoning Tiny Recursion Model (TRM) for Apple Silicon using MLX.  
Original CUDA/NVIDIA code can be found in [SamsungSAILMontreal/TinyRecursiveModels](https://github.com/SamsungSAILMontreal/TinyRecursiveModels).

***

## Project Overview

TinyRecursiveModels-MLX provides an efficient, small recursive neural net (TRM), ported to run natively on Mac Apple Silicon with [MLX](https://github.com/ml-explore/mlx).  
Designed for tasks like ARC-AGI, Sudoku, and maze puzzles with minimal parameters and high reasoning capability.

***

## Features

- ✅ **Native Apple Silicon/MLX Support** (no CUDA/NVIDIA required)
- ✅ **Test-Driven Codebase** for reliability and rapid iteration
- ✅ **Reproducible Mini-Datasets** for fast experimentation
- ✅ **Standard Benchmarks, Logging, and CLI**
- ✅ **Clear Documentation & Community Contribution Guidelines**

**Note:** Experimental or advanced features from the original repo are deferred or marked as optional for initial MLX release.

***

## Installation

### Prerequisites

- **Hardware:** Apple Silicon (M1, M2, M3, M4 series) 
- **OS:** macOS ≥ 13.5 (Ventura), recommended Sequoia/Sonoma
- **Python:** Native ARM build (Python ≥3.9, recommended 3.10+)
- **Memory:** At least 8GB RAM (16GB+ recommended for larger models)

Check your architecture:
```python
import platform; print(platform.processor())
# Should output: arm
```

### Setup Steps

1. **Clone this repo:**
    ```bash
    git clone https://github.com/gajsanders/TinyRecursiveModels-MLX.git
    cd TinyRecursiveModels-MLX
    ```

2. **Create virtual environment (recommended):**
    ```bash
    python -m venv mlx_env
    source mlx_env/bin/activate  # On macOS/Linux
    # Or for Windows: mlx_env\\Scripts\\activate
    ```

3. **Install MLX and dependencies:**  
    (Choose one)
    ```bash
    pip install mlx numpy pytest black isort flake8 mypy
    ```
    or using Conda:
    ```bash
    conda install -c conda-forge mlx numpy pytest black isort flake8 mypy
    ```

4. **Install project package:**  
    ```bash
    pip install -e .
    # Use -e flag to install in development mode for live changes
    ```

### Device Limitations

- **GPU Memory Constraints:** Apple Silicon GPUs have unified memory architecture with system RAM. Large models may exhaust available memory.
- **Precision Support:** MLX supports various floating point precisions (float16, float32, bfloat16) but behavior may differ from CUDA implementations.
- **Compute Limits:** Apple Silicon Neural Engine is not directly used by MLX; all computation runs on GPU/CPU.
- **Batch Size Considerations:** Due to memory limitations, you may need to reduce batch sizes compared to CUDA implementations.
- **No Distributed Training:** The MLX implementation currently does not support distributed training across multiple devices.

***

## Running the Model

### Basic Usage

Basic example of running a model forward pass:
```python
from trm_ml.model_trm import TRM
import numpy as np

model = TRM(input_dim=4, latent_dim=8, output_dim=1)
x = np.random.rand(2, 4)
y_pred = model.forward(x)
print("Predicted output:", y_pred)
```

### Command Line Interface

The project provides a comprehensive CLI for training, evaluation, and benchmarking:

```bash
# Training
python mlx/cli.py --train --model-input-dim 128 --model-latent-dim 256 --model-output-dim 64 --epochs 10 --batch-size 32

# Evaluation
python mlx/cli.py --eval --model-input-dim 128 --model-latent-dim 256 --model-output-dim 64 --batch-size 32

# Benchmarking
python mlx/cli.py --benchmark --model-input-dim 128 --model-latent-dim 256 --model-output-dim 64

# Or run the integrated pipeline
python mlx/wire_up.py --input-dim 128 --latent-dim 256 --output-dim 64 --epochs 3 --batch-size 16
```

***

## Testing

Run all tests to verify environment and code:
```bash
pytest tests/
```

For more specific testing:
```bash
# Run specific test file
pytest tests/test_model_trm.py

# Run with verbose output
pytest -v tests/

# Run with coverage report
pytest --cov=mlx/ tests/
```

***

## Benchmarking

### Running Benchmarks

Run the benchmark script to evaluate performance:
```bash
# Basic benchmark
python benchmarks/benchmark_trm.py

# For detailed benchmarks with specific parameters
python mlx/cli.py --benchmark --model-input-dim 256 --model-latent-dim 512 --model-output-dim 128
```

### Benchmarking Considerations

- **Model Size:** Larger models may not fit in Apple Silicon GPU memory
- **Batch Size Impact:** Performance varies significantly with different batch sizes
- **Memory Management:** MLX automatically manages memory allocation and deallocation
- **Precision Variations:** Different floating-point precisions may affect both performance and accuracy

***

## Directory Structure

- **mlx/** - MLX-native model and utilities
- **src/** - (Optional) legacy/shared code
- **tests/** - Full pytest suite
- **benchmarks/** - Benchmarking tools/scripts
- **examples/** - Jupyter notebooks and demo scripts
- **docs/** - Documentation and migration guides
- **legacy/** - Original CUDA/PyTorch code (see Legacy Code Notice below)
- **.github/workflows/** - Continuous integration configs

***

## Data

- Sample datasets for unit/integration tests are in `tests/data/`.
- For larger/open datasets, see included data fetch instructions or scripts.

**No private or proprietary datasets are bundled.**

***

## MLX Migration Notes

- All CUDA/Torch code paths are replaced or removed for Mac MLX.
- All device handling is Apple Silicon-aware.
- Advanced experimental features (e.g., HRM, distributed training) may be deferred; focus on core TRM reliability and testing.

***

## Contributing

We welcome contributions!  
Please:
- Write tests first (test-driven workflow)
- Keep code style and type hints consistent (`black`, `flake8`, `isort`, `mypy`)
- Submit PRs with clear doc/comments and test coverage
- See [CONTRIBUTING.md](CONTRIBUTING.md) for details

***

## License & Attribution

Project is MIT licensed – see [LICENSE](LICENSE).  
- Original authors, contributors, and all MLX source materials are credited.
- Please cite properly and follow usage guidelines for open datasets.

***

## Support & Questions

- Open an [issue](https://github.com/gajsanders/TinyRecursiveModels-MLX/issues) for bugs, questions, or feature requests.
- Join discussions on porting, benchmarking, or expanding features.

***

## References

- [MLX Docs](https://github.com/ml-explore/mlx)
- [Original Paper (arXiv:2510.04871)](https://arxiv.org/abs/2510.04871)
- [TinyRecursiveModels (Original CUDA Repo)](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)

***

**Less is more: Small models, deep reasoning, Mac MLX performance. Enjoy!**

Legacy Code Notice
The legacy/ directory contains original CUDA/PyTorch code and datasets from the source TinyRecursiveModels implementation.
These files are preserved for reference, incremental porting, and compatibility review, but are not MLX-native and will not run on Apple Silicon without major modification.

Only code inside the mlx/ directory and related MLX-native folders (e.g., benchmarks/, tests/, examples/) is supported on Mac MLX and actively maintained.
As the MLX migration progresses, features and logic will be refactored out of legacy/ and replaced with fully testable, Mac Apple Silicon-compatible modules.

If you need legacy CUDA functionality or want to compare old and new architectures, consult the files in legacy/, but please use mlx/ for all new work and contributions.

---
