import trm_ml.core as mx


class TRM:
    """
    Tiny Recursive Model (TRM) class
    """
    
    def __init__(self, input_dim, latent_dim, output_dim):
        """
        Initialize the TRM model.
        
        Args:
            input_dim (int): Dimension of input features
            latent_dim (int): Dimension of latent state
            output_dim (int): Dimension of output
            
        Raises:
            TypeError: If any dimension is not an integer
            ValueError: If any dimension is not positive
        """
        # Type checking
        if not isinstance(input_dim, int):
            raise TypeError(f"input_dim must be an integer, got {type(input_dim).__name__}")
        if not isinstance(latent_dim, int):
            raise TypeError(f"latent_dim must be an integer, got {type(latent_dim).__name__}")
        if not isinstance(output_dim, int):
            raise TypeError(f"output_dim must be an integer, got {type(output_dim).__name__}")
        
        # Value checking
        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {input_dim}")
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {output_dim}")
        
        # Store attributes
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
    
    def forward(self, x):
        """
        Forward pass of the TRM model.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim) filled with zeros
        """
        batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
        # Return zeros of shape (batch, output_dim) using MLX
        return mx.zeros((batch_size, self.output_dim))
    
    def initialize_state(self, batch_size):
        """
        Initialize the latent state for the TRM model.
        
        Args:
            batch_size (int): The size of the batch
            
        Returns:
            Latent state tensor of shape (batch_size, latent_dim) filled with zeros
        """
        # Return zeros of shape (batch_size, latent_dim) using MLX
        return mx.zeros((batch_size, self.latent_dim))
    
    def update_state(self, state, x, y):
        """
        Update the latent state of the TRM model.
        
        Args:
            state: Current state tensor
            x: Input tensor
            y: Target tensor (not used in this implementation)
            
        Returns:
            Updated state tensor
        """
        # For now, just return state + x.mean(axis=1, keepdims=True)
        return state + x.mean(axis=1, keepdims=True)
    
    def recursive_reasoning(self, x, y, steps):
        """
        Perform recursive reasoning by updating the state multiple times.
        
        Args:
            x: Input tensor
            y: Target tensor
            steps (int): Number of steps to run the update process
            
        Returns:
            Final state after running updates for the specified number of steps
        """
        # Get the batch size from the input x
        batch_size = x.shape[0] if hasattr(x, 'shape') else len(x)
        
        # Initialize the state using initialize_state method
        state = self.initialize_state(batch_size)
        
        # Run update_state steps times, each time updating the state
        for _ in range(steps):
            state = self.update_state(state, x, y)
        
        # Return the final state
        return state