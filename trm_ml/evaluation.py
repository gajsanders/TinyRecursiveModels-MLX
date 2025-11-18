import trm_ml.core as mx


def run_evaluation(model, data):
    """
    Run evaluation on the model using the provided data.

    Args:
        model: The TRM model to evaluate
        data: Evaluation data, expected to be an iterable of input tensors or (inputs, targets) tuples

    Returns:
        float: The mean output across all evaluation batches
    """
    all_outputs = []

    # Loop over batches in the data
    for batch_idx, batch in enumerate(data):
        # Handle both (inputs, targets) format and just inputs format
        if isinstance(batch, tuple) and len(batch) >= 2:
            inputs, _ = batch  # Extract inputs, ignore targets for evaluation
        else:
            inputs = batch  # The batch is just the inputs

        # Call model.forward() to get predictions/outputs
        outputs = model.forward(inputs)

        # Store outputs for calculating the mean
        # Convert to numpy for easier mean calculation if needed
        if hasattr(outputs, "tolist"):
            batch_outputs = outputs.tolist()
        elif hasattr(outputs, "item"):
            # If it's a scalar, just get the value
            batch_outputs = [outputs.item()]
        else:
            # For other formats, we'll handle appropriately
            batch_outputs = outputs

        # If it's a tensor/array, flatten and collect values
        if hasattr(batch_outputs, "__iter__") and not isinstance(
            batch_outputs, (str, bytes)
        ):
            # Flatten nested structures
            def flatten_to_numbers(obj):
                """Recursively flatten nested iterables to a list of numbers."""
                result = []
                if isinstance(obj, (list, tuple)):
                    for item in obj:
                        result.extend(flatten_to_numbers(item))
                elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes)):
                    # Handle array-like objects
                    try:
                        for item in obj:
                            result.extend(flatten_to_numbers(item))
                    except TypeError:
                        # If we can't iterate, treat as scalar
                        try:
                            result.append(float(obj))
                        except (TypeError, ValueError):
                            pass
                else:
                    # Handle scalar values
                    try:
                        result.append(float(obj))
                    except (TypeError, ValueError):
                        pass
                return result

            flattened = flatten_to_numbers(batch_outputs)
            all_outputs.extend(flattened)
        else:
            # Handle scalar values
            try:
                all_outputs.append(float(batch_outputs))
            except (TypeError, ValueError):
                pass

    # Calculate and return the mean of all outputs
    if all_outputs:
        # Convert to mlx array and compute mean
        all_outputs_array = mx.array(all_outputs)
        mean_output = mx.mean(all_outputs_array)
        # Return as a Python float
        if hasattr(mean_output, "item"):
            return mean_output.item()
        else:
            try:
                return float(mean_output)
            except Exception:
                # Last resort fallback
                return 0.0
    else:
        # Return 0.0 if no data was processed
        return 0.0
