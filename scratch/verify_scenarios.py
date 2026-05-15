import sys
import os
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fpl_engine.scenarios import generate_scenario_tensor

# Mock data
np.random.seed(42)
players = [1, 2]
gws = [1, 2]
n_scenarios = 50

df = pd.DataFrame({
    'id_player': [1, 1, 2, 2],
    'gameweek': [1, 2, 1, 2],
    'Perf_IDX': [5.0, 6.0, 3.0, 4.0],
    'score_std': [2.0, 2.5, 1.5, 2.0]
})

corr_matrix = np.eye(6)
component_corr = {'corr_matrix': corr_matrix}

try:
    tensor = generate_scenario_tensor(df, component_corr, n_scenarios=n_scenarios, seed=42)
    print(f"Tensor shape: {tensor.shape}")
    if tensor.shape == (2, 2, 50):
        print("Success: Shape is correct.")
    else:
        print("Error: Incorrect shape.")
        
    # Check independence between players for the same gameweek and scenario
    p1_gw1 = tensor[0, 0, :]
    p2_gw1 = tensor[1, 0, :]
    
    correlation = np.corrcoef(p1_gw1, p2_gw1)[0, 1]
    print(f"Correlation between P1 and P2 in GW1: {correlation:.4f}")
    
    if abs(correlation) < 0.3:
        print("Success: Players have independent random draws.")
    else:
        print("Warning: Players seem too highly correlated.")
        
except Exception as e:
    print(f"Error: {e}")
