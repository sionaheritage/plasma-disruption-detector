import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

from data_generator import generate_synthetic_plasma_data
from lstm_model import PlasmaDisruptionPredictor

def run_evaluation():
    # -----%-----
    # 1. SETUP AND MODEL LOADING
    # -----%-----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on: {device}")
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, '../models/best_plasma_lstm.pth')
    
    model = PlasmaDisruptionPredictor().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    # -----%-----
    # 2. STAT EVALUATION (Test Set)
    # -----%-----
    print("\nGenerating 1,000 unseen test samples...")
    X_test, y_test = generate_synthetic_plasma_data(num_samples=1000, seq_length=50)
    X_test, y_test = X_test.to(device), y_test.to(device)
    
    with torch.no_grad():
        predictions = model(X_test).squeeze()
        predicted_classes = (predictions > 0.5).float()
    
    y_true = y_test.cpu().numpy()
    y_pred = predicted_classes.cpu().numpy()
    
    print("\n--- Model Performance Metrics ---")
    # This generates a professional table showing Precision, Recall, and F1-Score
    print(classification_report(y_true, y_pred, target_names=['Stable', 'Disruptive']))
    
    # -----%-----
    # 3. VISUALISING DISRUPTION EVENT
    # -----%-----
    print("\nGenerating visualization for a disruptive event...")
    
    # Find a sample in test set that is an actual disruption (Label == 1)
    disruptive_idx = np.where(y_true == 1)[0][0]
    X_sample = X_test[disruptive_idx:disruptive_idx+1] # Shape: (1, 50, 5)
    
    # Feed the sequence to the model incrementally to get a "live" risk score
    # Start at time step 10 to give the LSTM a baseline memory
    time_steps = range(10, 51)
    risk_scores = []
    
    with torch.no_grad():
        for t in time_steps:
            # Slicing the sequence: up to time step 't'
            current_sequence = X_sample[:, :t, :]
            prob = model(current_sequence).item()
            risk_scores.append(prob * 100) # Convert to percentage
            
    # -----%-----
    # PLOTTING 
    # -----%-----
    X_numpy = X_sample.squeeze().cpu().numpy()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # 1. Physics Telemetry
    ax1.set_title("Tokamak Telemetry & AI Disruption Prediction", fontsize=14, fontweight='bold')
    ax1.plot(X_numpy[:, 2], label="Plasma Beta (Pressure)", color='orange')
    ax1.plot(X_numpy[:, 3], label="Safety Factor (q)", color='green')
    ax1.plot(X_numpy[:, 4], label="MHD Amplitude", color='red')
    ax1.axvline(x=30, color='grey', linestyle='--', label="Instability Onset")
    ax1.set_ylabel("Normalized Sensor Values")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    
    # 2. AI Risk Score
    ax2.plot(time_steps, risk_scores, color='darkred', linewidth=2, label="AI Risk Score")
    ax2.axhline(y=85, color='black', linestyle=':', label="Actuation Threshold (85%)")
    ax2.fill_between(time_steps, risk_scores, 85, where=(np.array(risk_scores) >= 85), 
                     color='red', alpha=0.3, label="Critical Zone")
    ax2.set_xlabel("Time Steps (Milliseconds)")
    ax2.set_ylabel("Disruption Probability (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="upper left")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(BASE_DIR, '../media/disruption_analysis.png')
    plt.savefig(save_path, dpi=300)
    print(f"Visualization saved successfully to: {save_path}")

if __name__ == "__main__":
    run_evaluation()