Here is a **new README.md** tailored for your TinyRecursiveModels-MLX MLX/Apple Silicon port, reflecting the current project goals, setup instructions, and contribution guidelines:

***

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

## Quickstart (Apple Mac Only)

### Prerequisites

- **Hardware:** Apple Silicon (M1, M2, M3, M4 series)
- **OS:** macOS ≥ 13.5 (Ventura), recommended Sequoia/Sonoma
- **Python:** Native ARM build (Python ≥3.9, recommended 3.10+)

Check your architecture:
```python
import platform; print(platform.processor())
# Should output: arm
```

### Installation

1. **Clone this repo:**
    ```bash
    git clone https://github.com/gajsanders/TinyRecursiveModels-MLX.git
    cd TinyRecursiveModels-MLX
    ```

2. **Install MLX and dependencies:**  
    (Choose one)
    ```bash
    pip install mlx numpy pytest black isort flake8 mypy
    ```
    or using Conda:
    ```bash
    conda install -c conda-forge mlx numpy pytest black isort flake8 mypy
    ```

3. **Install project package:**  
    ```bash
    pip install .
    ```

### Testing

Run all tests to verify environment and code:
```bash
pytest tests/
```

***

## Usage

Basic example of running a model forward pass:
```python
from mlx.model_trm import TRM
import numpy as np

model = TRM(input_dim=4, latent_dim=8, output_dim=1)
x = np.random.rand(2, 4)
y_pred = model.forward(x)
print("Predicted output:", y_pred)
```

Run benchmark script:
```bash
python benchmarks/benchmark_trm.py
```

Train and evaluate (see CLI for options):
```bash
python mlx/cli.py --train
python mlx/cli.py --eval
```

***

## Directory Structure

- **mlx/** - MLX-native model and utilities
- **src/** - (Optional) legacy/shared code
- **tests/** - Full pytest suite
- **benchmarks/** - Benchmarking tools/scripts
- **examples/** - Jupyter notebooks and demo scripts
- **docs/** - Documentation and migration guides
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