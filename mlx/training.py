import mlx.core as mx


def train_one_epoch(model, data, optimizer):
    """
    Train the model for one epoch.
    
    Args:
        model: The TRM model to train
        data: Training data, expected to be an iterable of (inputs, targets) tuples
        optimizer: Optimization algorithm (for now, just a stub that prints losses)
    
    Returns:
        float: Average loss for the epoch
    """
    total_loss = 0.0
    num_batches = 0
    
    # Loop over batches in the data
    for batch_idx, (inputs, targets) in enumerate(data):
        # Call model.forward() to get predictions
        predictions = model.forward(inputs)
        
        # Compute mean squared error with target
        # MSE = mean((predictions - targets)^2)
        squared_errors = (predictions - targets) ** 2
        batch_loss = mx.mean(squared_errors)
        
        # Add the batch loss to the total
        total_loss += batch_loss.item() if hasattr(batch_loss, 'item') else float(batch_loss)
        num_batches += 1
        
        # For now, stub optimizer and just print losses
        print(f"Batch {batch_idx + 1}, Loss: {batch_loss}")
        
        # Call optimizer.step() - in this stub implementation, we're just printing
        # In a full implementation, this would update the model parameters
        # optimizer.step(batch_loss) # This would be the actual call
    
    # Calculate average loss for the epoch
    if num_batches > 0:
        avg_loss = total_loss / num_batches
        print(f"Epoch completed. Average Loss: {avg_loss}")
        return avg_loss
    else:
        print("No batches processed.")
        return 0.0