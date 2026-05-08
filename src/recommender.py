import numpy as np

def suggest_adjustments(recent_data, disruption_prob, threshold=0.85):
    """
    Analyzes the most recent plasma state and recommends physical control adjustments
    to avert a predicted disruption.
    
    Args:
        recent_data (np.ndarray): The recent sequence of plasma telemetry, typically shape (seq_length, 5).
        disruption_prob (float): The LSTM's predicted probability of disruption (0.0 to 1.0).
        threshold (float): The probability limit above which evasive action is taken.
        
    Returns:
        str: A formatted string containing system directives.
    """
    # 1. Check if the system is stable
    if disruption_prob < threshold:
        return "[STATUS: NOMINAL] System Stable. Maintain current magnetic confinement parameters."
        
    # 2. Extract the current state (the very last time step in the sequence)
    # Feature Index mapping based on data_generator.py: 
    # [0: Ip, 1: Bt, 2: Beta, 3: q, 4: MHD]
    latest_state = recent_data[-1] 
    Ip, Bt, beta, q, mhd = latest_state
    
    recommendations = ["[WARNING] DISRUPTION IMMINENT! Initiating mitigation protocols:"]
    
    # -----%-----
    # 3. HEURISTICS+ACTUATION
    # -----%-----
    
    # Mitigation A: Beta Limit Exceeded (Pressure-driven ballooning modes)
    if beta > 0.035:
        recommendations.append(
            " -> ACTUATION: Reduce auxiliary heating (NBI/ECRH power) to lower Plasma Beta."
        )
        
    # Mitigation B: Low Safety Factor (Current-driven kink/tearing modes)
    if q < 2.5:
        recommendations.append(
            " -> ACTUATION: Ramp down plasma current (Ip) or increase Toroidal Field (Bt) to raise q-profile."
        )
        
    # Mitigation C: High MHD Fluctuations (Growing magnetic islands)
    if mhd > 0.05:
        recommendations.append(
            " -> ACTUATION: Steer Electron Cyclotron Resonance Heating (ECRH) to the q=2 rational surface to stabilize tearing modes."
        )
        
    # Mitigation D: Point of No Return (Extreme risk or unknown cause)
    if disruption_prob > 0.96 or len(recommendations) == 1:
        recommendations.append(
            " -> CRITICAL: Trigger Massive Gas Injection (MGI) valves to safely quench plasma and protect the divertor walls."
        )
        
    return "\n".join(recommendations)

if __name__ == "__main__":
    # Quick test to ensure logic triggers correctly
    print("Testing Recommender Logic...\n")
    
    # Dummy data representing an unstable state at the last time step
    # [Ip=1.0, Bt=2.5, Beta=0.04 (High), q=2.1 (Low), MHD=0.08 (High)]
    dummy_sequence = np.zeros((50, 5))
    dummy_sequence[-1] = [1.0, 2.5, 0.04, 2.1, 0.08] 
    
    # Simulate a 92% disruption prediction from the LSTM
    output = suggest_adjustments(dummy_sequence, disruption_prob=0.92)
    print(output)