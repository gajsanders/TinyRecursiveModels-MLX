import random

import trm_ml.core as mx


def get_device():
    """Helper function to get the available device."""
    try:
        return "mlx"
    except ImportError:
        return "cpu"


class DataLoader:
    """
    A data loader that shuffles data, creates batches, and moves them to the target device.
    """

    def __init__(self, inputs, targets, batch_size=32, shuffle=True, device=None):
        """
        Initialize the data loader.

        Args:
            inputs: Input data (should be array-like)
            targets: Target data (should be array-like)
            batch_size (int): Size of each batch
            shuffle (bool): Whether to shuffle the data
            device: Target device, if None will auto-detect
        """
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.device = device or get_device()

        # Ensure inputs and targets have the same length
        if len(inputs) != len(targets):
            raise ValueError(
                f"Inputs and targets must have the same length. "
                f"Got {len(inputs)} inputs and {len(targets)} targets."
            )

        # Create indices for shuffling
        self.indices = list(range(len(inputs)))

    def __iter__(self):
        """
        Create an iterator that yields batches of shuffled, batched data on the target device.
        """
        # Shuffle indices if requested
        if self.shuffle:
            # Use both random and numpy for better randomization
            shuffled_indices = self.indices[:]
            random.shuffle(shuffled_indices)
        else:
            shuffled_indices = self.indices[:]

        # Yield batches
        for i in range(0, len(shuffled_indices), self.batch_size):
            batch_indices = shuffled_indices[i : i + self.batch_size]

            # Get the actual data for this batch
            batch_inputs = [self.inputs[idx] for idx in batch_indices]
            batch_targets = [self.targets[idx] for idx in batch_indices]

            # Convert to mlx arrays and move to device if needed
            if self.device == "mlx":
                # Convert to mlx arrays - ensure compatibility with MLX's array creation
                import numpy as np

                batch_inputs = mx.array(np.array(batch_inputs))
                batch_targets = mx.array(np.array(batch_targets))

            yield batch_inputs, batch_targets

    def __len__(self):
        """
        Return the number of batches.
        """
        return (
            len(self.indices) + self.batch_size - 1
        ) // self.batch_size  # Ceiling division


def create_dataloader(inputs, targets, batch_size=32, shuffle=True, device=None):
    """
    Convenience function to create a DataLoader.

    Args:
        inputs: Input data (should be array-like)
        targets: Target data (should be array-like)
        batch_size (int): Size of each batch
        shuffle (bool): Whether to shuffle the data
        device: Target device, if None will auto-detect

    Returns:
        DataLoader: A configured data loader instance
    """
    return DataLoader(inputs, targets, batch_size, shuffle, device)
