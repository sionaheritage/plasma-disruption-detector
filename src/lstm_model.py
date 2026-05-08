import torch
import torch.nn as nn

class PlasmaDisruptionPredictor(nn.Module):
    """
    Long Short-Term Memory (LSTM) Neural Network for time-series anomaly detection.
    Predicts the probability of an imminent plasma disruption based on telemetry history.
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2, dropout_rate=0.2):
        super(PlasmaDisruptionPredictor, self).__init__()
        
        # 1. The LSTM Layer
        # input_size: The number of features (5 telemetry parameters)
        # hidden_size: The number of neurons in the LSTM's memory cells
        # num_layers: Stacking 2 LSTMs allows the model to learn more complex patterns
        # batch_first=True: Expects input tensors shaped as (batch_size, sequence_length, features)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # 2. Fully Connected (Dense) Layers
        # Maps the 64-dimensional LSTM output down to a 32-dimensional space
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU() # Non-linear activation
        
        # Maps the 32-dimensional space down to a single output node
        self.fc2 = nn.Linear(32, 1)
        
        # 3. Output Activation
        # Squashes the final output into a probability between 0.0 and 1.0
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Defines the forward pass of the data through the network.
        x shape: (batch_size, sequence_length, features)
        """
        # Pass data through the LSTM
        # lstm_out shape: (batch_size, seq_length, hidden_size)
        lstm_out, (hidden_state, cell_state) = self.lstm(x)
        
        # We only care about the network's output at the VERY LAST time step 
        # of the sequence to make our prediction.
        last_step_out = lstm_out[:, -1, :] 
        
        # Pass through the fully connected layers
        out = self.fc1(last_step_out)
        out = self.relu(out)
        out = self.fc2(out)
        
        # Return the final probability
        return self.sigmoid(out)

if __name__ == "__main__":
    # Quick test to ensure the model compiles and tensor shapes align
    print("Initializing LSTM Model...")
    model = PlasmaDisruptionPredictor()
    
    # Create a dummy tensor representing a single batch of 1 sequence, 
    # 50 time steps long, with 5 features.
    dummy_input = torch.randn(1, 50, 5)
    
    # Run a forward pass
    output_prob = model(dummy_input)
    
    print(f"Model successfully processed input. Output shape: {output_prob.shape}")
    print(f"Sample Disruption Probability: {output_prob.item() * 100:.2f}%")