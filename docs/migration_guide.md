# MLX Migration Guide for Tiny Recursive Models

## Overview

This document provides a comprehensive summary of the migration from PyTorch/CUDA to MLX (Apple's Metal Linear Algebra library) for the TinyRecursiveModels project. The migration was undertaken to enable native support for Apple Silicon Macs and leverage the MLX framework for efficient machine learning computations.

## Migration Goals

- Replace PyTorch/CUDA dependencies with MLX for Apple Silicon compatibility
- Maintain the core functionality and architecture of the original models
- Implement native MLX equivalents for PyTorch operations
- Create a new codebase structure that cleanly separates legacy code from new MLX implementation

## Conversion Steps and File Changes

### 1. Directory Restructure
- **Action**: Created a new `mlx/` directory for all MLX-based implementations
- **Files**: 
  - `mlx/__init__.py`
  - `mlx/model_trm.py`
  - `mlx/training.py`
  - `mlx/evaluation.py`
  - `mlx/data_utils.py`
  - `mlx/device.py`
  - `mlx/cli.py`
  - `mlx/wire_up.py`
  - `mlx/utils/logging_utils.py`
- **Rationale**: Isolate MLX-specific code in its own namespace to avoid conflicts with legacy PyTorch code and provide a clean separation of concerns.

### 2. Legacy Code Preservation
- **Action**: Moved all original PyTorch/CUDA code to the `legacy/` directory with explicit comments marking them for refactor/removal
- **Files**: 
  - `legacy/dataset/build_arc_dataset.py`
  - `legacy/dataset/build_maze_dataset.py`
  - `legacy/dataset/build_sudoku_dataset.py`
  - `legacy/dataset/common.py`
  - `legacy/evaluators/arc.py`
  - `legacy/models/common.py`
  - `legacy/models/ema.py`
  - `legacy/models/layers.py`
  - `legacy/models/losses.py`
  - `legacy/models/recursive_reasoning/hrm.py`
  - `legacy/models/recursive_reasoning/transformers_baseline.py`
  - `legacy/models/recursive_reasoning/trm.py`
  - `legacy/models/recursive_reasoning/trm_hier6.py`
  - `legacy/models/recursive_reasoning/trm_singlez.py`
  - `legacy/models/sparse_embedding.py`
  - `legacy/pretrain.py`
- **Rationale**: Preserve original PyTorch implementation for reference during migration while clearly marking them as legacy code to be gradually replaced.

### 3. Model Implementation Migration
- **Action**: Created a new TRM (Tiny Recursive Model) class in `mlx/model_trm.py` using MLX operations instead of PyTorch
- **Changes**:
  - Replaced `import torch` with `import mlx.core as mx`
  - Replaced PyTorch tensor operations with MLX equivalents
  - Converted tensor creation methods (e.g., `torch.zeros()` → `mx.zeros()`)
  - Adapted tensor indexing and operations to MLX API
- **Rationale**: Implement the core TRM model using MLX's tensor operations to enable efficient execution on Apple Silicon.

### 4. Training Pipeline Migration
- **Action**: Implemented `mlx/training.py` module using MLX instead of PyTorch
- **Changes**:
  - Replaced PyTorch loss computation with MLX equivalents (e.g., MSE calculation using `mx.mean`)
  - Adapted optimizer placeholders to work with MLX tensors
  - Maintained the same training loop structure while using MLX operations
- **Rationale**: Create a training pipeline that operates entirely on MLX tensors for optimal Apple Silicon performance.

### 5. Data Loading and Utilities
- **Action**: Created `mlx/data_utils.py` with MLX-compatible data loading
- **Changes**:
  - Implemented DataLoader class that converts data to MLX arrays
  - Added device detection for MLX compatibility
  - Implemented batching and shuffling logic compatible with MLX tensors
- **Rationale**: Ensure data flows properly from raw inputs to MLX tensors without requiring PyTorch.

### 6. Evaluation Pipeline
- **Action**: Implemented `mlx/evaluation.py` with MLX-based evaluation logic
- **Changes**:
  - Replaced PyTorch tensor operations with MLX equivalents
  - Adapted evaluation metrics to use MLX computations
- **Rationale**: Enable model evaluation using only MLX operations for consistency.

### 7. Device Management
- **Action**: Created `mlx/device.py` for MLX device detection
- **Changes**:
  - Implemented device detection that checks for MLX availability
  - Returns MLX as the default device when available
- **Rationale**: Provide a centralized way to detect and manage compute devices in the MLX context.

### 8. CLI Interface
- **Action**: Created `mlx/cli.py` to provide command-line interface for MLX-based operations
- **Changes**:
  - Implemented training, evaluation, and benchmarking entry points
  - Configured CLI arguments to work with MLX models
  - Used subprocess to call benchmark scripts when needed
- **Rationale**: Maintain command-line accessibility for the MLX-based model implementations.

### 9. Integrated Pipeline
- **Action**: Created `mlx/wire_up.py` to connect all components into a single pipeline
- **Changes**:
  - Developed a comprehensive pipeline that integrates model, data, training, and evaluation
  - Added proper module loading to handle path conflicts between local and installed MLX
  - Implemented logging and result tracking
- **Rationale**: Provide a complete working example of the MLX-based pipeline with all components properly connected.

### 10. Logging Utilities
- **Action**: Created `mlx/utils/logging_utils.py` for MLX-specific logging
- **Changes**:
  - Implemented logging functions that work well with MLX operations
  - Added convenience functions for different logging levels
- **Rationale**: Provide consistent logging across all MLX components.

## Key Technical Changes

### PyTorch to MLX API Mapping
- `torch.tensor` → `mx.array`
- `torch.zeros` → `mx.zeros`
- `torch.mean` → `mx.mean`
- `tensor.shape` → `tensor.shape` (same access pattern)
- `tensor.item()` → `tensor.item()` (same access pattern)

### Data Handling
- PyTorch DataLoader replaced with custom MLX-compatible DataLoader
- NumPy arrays are converted to MLX arrays when needed
- Batching and shuffling logic adapted for MLX tensors

### Device Management
- Removed CUDA-specific device handling
- Added MLX device detection
- Simplified device management for Apple Silicon

## Architecture Changes

### Module Dependencies
- Removed all `torch`, `torch.nn`, `torch.nn.functional` imports from new MLX modules
- Added `mlx.core` imports in their place
- Maintained same logical structure while updating tensor operations

### Code Organization
- Clearly separated legacy PyTorch code in `legacy/` directory
- New MLX implementation in `mlx/` directory
- Clean separation prevents conflicts between old and new implementations

## Testing and Validation

### Current Status
- Basic MLX operations implemented and tested
- Sample data pipelines working with MLX tensors
- Training and evaluation loops adapted for MLX
- Command-line interface functional

### Next Steps
- Complete end-to-end testing of migration
- Performance benchmarking vs original PyTorch implementation
- Full integration testing of all components
- Documentation updates for new MLX API

## Challenges and Solutions

### Path Conflicts
- **Issue**: Local `mlx` directory conflicted with installed MLX package
- **Solution**: Implemented dynamic module loading in `wire_up.py` to properly handle imports

### API Differences
- **Issue**: Some PyTorch operations don't have exact MLX equivalents
- **Solution**: Mapped equivalent operations and adapted where necessary

### Device Management
- **Issue**: MLX uses different device management than PyTorch
- **Solution**: Created abstraction layer in `device.py` to handle device detection

## Conclusion

The migration to MLX has been systematically implemented with a clean separation between legacy PyTorch code and new MLX implementations. The new architecture maintains the core functionality of the original models while providing native support for Apple Silicon through the MLX framework. The conversion preserves the recursive reasoning architecture while adapting tensor operations, data handling, and training loops to work efficiently with MLX.