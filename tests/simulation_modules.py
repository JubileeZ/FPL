import sys
import os
import pandas as pd
import numpy as np

# Ensure fpl_engine is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpl_engine.scoring import _estimate_component_correlation
from fpl_engine.scenarios import generate_scenario_tensor, compute_squad_cvar
from tests.synthetic_data import generate_synthetic_history, generate_synthetic_projections

def test_covariance_estimator():
    print("\n--- Testing Covariance Estimator ---")
    history = generate_synthetic_history(n_players=100, n_fixtures=20)
    
    result = _estimate_component_correlation(history)
    corr_matrix = result['corr_matrix']
    
    print(f"Empirical Correlation Matrix (Sample Size: {result['n_observations']}):")
    print(pd.DataFrame(corr_matrix, columns=result['component_order'], index=result['component_order']).round(3))
    
    # Assertions
    assert corr_matrix.shape == (6, 6)
    assert np.all(np.diag(corr_matrix) >= 0.99) # Diagonals should be ~1.0
    assert np.all(corr_matrix <= 1.0)
    print("✅ Covariance Estimator passed basic validity checks.")

def test_scenario_tensor_convergence():
    print("\n--- Testing Scenario Tensor Convergence ---")
    projections = generate_synthetic_projections(n_players=10, n_gw=3)
    history = generate_synthetic_history(n_players=100, n_fixtures=20)
    corr_result = _estimate_component_correlation(history)
    
    n_scenarios = 10000
    tensor = generate_scenario_tensor(projections, corr_result, n_scenarios=n_scenarios)
    
    # Check shape: (players, gw, scenarios)
    # projections has 10 players, 3 GW
    assert tensor.shape == (10, 3, n_scenarios)
    
    # Check mean convergence
    player_id = 1
    gw_idx = 0
    expected_mean = projections[(projections['id_player'] == player_id) & (projections['gameweek'] == 1)]['Perf_IDX'].iloc[0]
    actual_mean = np.mean(tensor[0, gw_idx, :])
    
    error = abs(actual_mean - expected_mean)
    print(f"Expected Mean: {expected_mean:.3f}, Actual Scenario Mean: {actual_mean:.3f} (Error: {error:.4f})")
    assert error < 0.1 # Should converge with 10k samples
    
    # Check CVaR monotonicity
    print("\n--- Testing CVaR Monotonicity ---")
    squad_indices = [0, 1, 2, 3, 4] # Top 5 players in tensor
    
    cvar_10 = compute_squad_cvar(tensor, squad_indices, captain_idx_in_squad=0, alpha=0.10)
    cvar_50 = compute_squad_cvar(tensor, squad_indices, captain_idx_in_squad=0, alpha=0.50)
    
    print(f"CVaR @ 10%: {cvar_10:.2f}")
    print(f"CVaR @ 50%: {cvar_50:.2f}")
    assert cvar_10 < cvar_50
    print("✅ CVaR Monotonicity confirmed.")

if __name__ == "__main__":
    test_covariance_estimator()
    test_scenario_tensor_convergence()
