import os
import yaml # NEW
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from data_generator import generate_synthetic_plasma_data
from lstm_model import PlasmaDisruptionPredictor

def load_config():
    """Loads the YAML configuration file."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '../config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train_model():
    # ==========================================
    # 1. Load Configurations
    # ==========================================
    config = load_config()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, config['paths']['model_dir'])
    model_path = os.path.join(model_dir, config['paths']['model_name'])
    
    # ==========================================
    # 2. Data Preparation
    # ==========================================
    print("Generating synthetic plasma data...")
    X, y = generate_synthetic_plasma_data(
        num_samples=config['data']['train_samples'], 
        seq_length=config['data']['seq_length']
    )
    
    dataset = TensorDataset(X, y)
    
    train_size = int(config['training']['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # ==========================================
    # 3. Model, Loss, Optimizer Setup
    # ==========================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Pass model configs dynamically
    model = PlasmaDisruptionPredictor(
        input_size=config['data']['num_features'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout_rate=config['model']['dropout_rate']
        
    ).to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    # -----%-----
    # 4. TRAINING LOOP
    # -----%-----
    best_val_loss = float('inf')
    os.makedirs('../models', exist_ok=True) # Ensure the models directory exists
    EPOCHS = config['training']['epochs']
    print("\nStarting Training...\n" + "-"*30)
    for epoch in range(EPOCHS):
        
        # --- Training Phase ---
        model.train() # Set model to training mode
        train_loss, correct_train, total_train = 0.0, 0, 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()              # Clear old gradients
            predictions = model(batch_X).squeeze() # Forward pass
            
            loss = criterion(predictions, batch_y) # Calculate error
            loss.backward()                    # Backpropagation
            optimizer.step()                   # Update weights
            
            # Track metrics
            train_loss += loss.item() * batch_X.size(0)
            predicted_classes = (predictions > 0.5).float()
            correct_train += (predicted_classes == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        avg_train_loss = train_loss / train_size
        train_acc = correct_train / total_train

        # --- Validation Phase ---
        model.eval() # Set model to evaluation mode
        val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad(): # Disable gradient calculation for speed/memory
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                predictions = model(batch_X).squeeze()
                
                loss = criterion(predictions, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
                predicted_classes = (predictions > 0.5).float()
                correct_val += (predicted_classes == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        avg_val_loss = val_loss / val_size
        val_acc = correct_val / total_val

        # --- Logging and Checkpointing ---
        # Print progress every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1:02d}/{EPOCHS}] | "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        # Save the model if it's the best one we've seen so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), model_path)

    print("\n" + "-"*30)
    print("Training complete! Best model weights saved to 'models/best_plasma_lstm.pth'")

if __name__ == "__main__":
    train_model()