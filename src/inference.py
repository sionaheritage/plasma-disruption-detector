import os
import yaml
import torch
import numpy as np

from lstm_model import PlasmaDisruptionPredictor
from recommender import suggest_adjustments
from data_generator import generate_synthetic_plasma_data

def load_config():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, '../config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def run_live_monitor():
    config = load_config()
    
    # Setup Paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, config['paths']['model_dir'], config['paths']['model_name'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialises Model dynamically
    model = PlasmaDisruptionPredictor(
        input_size=config['data']['num_features'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'],
        dropout_rate=config['model']['dropout_rate']
    ).to(device)

    # -----%-----
    # 3. SIMULATE "LIVE" TELEMETRY DATA
    # -----%-----
    print("Awaiting plasma diagnostics...")
    # Generate 1 sequence to represent the last 50 time steps in a reactor
    X_live, actual_label = generate_synthetic_plasma_data(num_samples=1, seq_length=50)
    X_live = X_live.to(device)

    # -----%-----
    # 4. PREDICTION
    # -----%-----
    with torch.no_grad(): 
        prediction_prob = model(X_live).item()

    # -----%-----
    # 5. RECOMMENDATION
    # -----%-----
    print("\n" + "-"*30)
    print("TOKAMAK REAL-TIME MONITOR!")
    print("-"*30)
    
    # Show the "Ground Truth" just so you can verify the model is accurate
    ground_truth = 'Disruptive' if actual_label.item() == 1 else 'Stable'
    print(f"Actual Plasma State:  {ground_truth}")
    print(f"AI Predicted Risk:    {prediction_prob * 100:.2f}%\n")
    
    # Convert the tensor back to a numpy array for the recommender logic
    recent_data_np = X_live.squeeze().cpu().numpy()
    recommendation = suggest_adjustments(
         recent_data_np, 
         prediction_prob, 
         threshold=config['inference']['actuation_threshold']
         )
    
    print("SYSTEM DIRECTIVE:")
    print("-" * 17)
    print(recommendation)
    print("="*45)

if __name__ == "__main__":
    run_live_monitor()