import numpy as np
import torch

def _generate_random_walk(length, start_val, volatility):
    """Helper function to generate realistic sensor noise/drift."""
    steps = np.random.normal(loc=0, scale=volatility, size=length)
    walk = start_val + np.cumsum(steps)
    return walk

def generate_synthetic_plasma_data(num_samples=1000, seq_length=50):
    """
    Generates synthetic time-series data for a Tokamak.
    Features (5): 
        0: Plasma Current (Ip) - [MA]
        1: Toroidal Magnetic Field (Bt) - [Tesla]
        2: Plasma Beta (beta) - Ratio of pressures
        3: Safety Factor (q) - Magnetic pitch
        4: MHD Amplitude - Magnetic fluctuation strength
    """
    # 0 = Stable, 1 = Disruption Imminent
    labels = np.random.randint(0, 2, num_samples)
    data = np.zeros((num_samples, seq_length, 5))
    
    for i in range(num_samples):
        # 1. Generate baseline stable telemetry using random walks
        Ip = _generate_random_walk(seq_length, start_val=1.0, volatility=0.01)
        Bt = _generate_random_walk(seq_length, start_val=2.5, volatility=0.005)
        beta = _generate_random_walk(seq_length, start_val=0.02, volatility=0.001)
        q = _generate_random_walk(seq_length, start_val=3.5, volatility=0.05)
        mhd = np.abs(np.random.normal(0, 0.01, seq_length)) # Baseline low-level noise
        
        # 2. Inject Physics-Based Instability Signatures
        if labels[i] == 1:
            disruption_window = 20
            
            # Add noise to the Beta increase
            beta_drift = np.linspace(0, 0.03, disruption_window)
            beta_noise = np.random.normal(0, 0.005, disruption_window)
            beta[-disruption_window:] += (beta_drift + beta_noise)
            
            # Add noise to the q-factor drop
            q_drop = np.linspace(0, 1.2, disruption_window)
            q_noise = np.random.normal(0, 0.15, disruption_window)
            q[-disruption_window:] -= (q_drop + q_noise)
            
            # Make the MHD spike less perfectly exponential
            growth_rate = 1.3
            mhd_spike = np.array([0.05 * (growth_rate ** j) for j in range(disruption_window)])
            mhd_noise = np.random.normal(0, 0.02, disruption_window)
            mhd[-disruption_window:] += np.abs(mhd_spike + mhd_noise)
            
        # 3. Stack features into shape (seq_length, num_features)
        data[i] = np.stack([Ip, Bt, beta, q, mhd], axis=-1)
        
    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

if __name__ == "__main__":
    # Quick test to ensure shape is correct
    X, y = generate_synthetic_plasma_data(num_samples=10, seq_length=50)
    print(f"Data Shape: {X.shape} (Samples, Sequence Length, Features)")
    print(f"Labels Shape: {y.shape}")