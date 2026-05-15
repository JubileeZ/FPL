import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

def generate_scenario_tensor(
    gw_projection_df: pd.DataFrame,
    component_corr: dict,
    n_scenarios: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    """
    Generates a (Players, Gameweeks, Scenarios) tensor of FPL point outcomes.
    Uses a Gaussian Copula to maintain joint correlation between scoring components.
    """
    np.random.seed(seed)
    
    # Unique players and gameweeks in the projection set
    players = gw_projection_df['id_player'].unique()
    gameweeks = sorted(gw_projection_df['gameweek'].unique())
    n_players = len(players)
    n_gw = len(gameweeks)
    
    # Initialize tensor
    tensor = np.zeros((n_players, n_gw, n_scenarios))
    
    # Map player_id to row index
    player_idx_map = {pid: i for i, pid in enumerate(players)}
    
    # R: Correlation Matrix (from scoring.py)
    R = component_corr.get("corr_matrix", np.eye(6))
    L = np.linalg.cholesky(R) # Cholesky decomposition for correlated normal draws
    
    # Process per GW
    for t_idx, gw in enumerate(gameweeks):
        gw_df = gw_projection_df[gw_projection_df['gameweek'] == gw]
        if gw_df.empty: continue
        
        means = gw_df['Perf_IDX'].values
        stds = gw_df['score_std'].values
        p_ids = gw_df['id_player'].values
        n_rows = len(gw_df)
        
        # 1. Draw correlated standard normals for the 6 components *per player*
        # We need (6, n_rows, n_scenarios) standard normals.
        # This ensures players have independent draws, avoiding 100% squad correlation.
        base_normals = np.random.normal(0, 1, (6, n_rows, n_scenarios))
        
        # Apply Cholesky decomposition L (shape 6,6) to the components
        # L @ base_normals => (6, 6) @ (6, n_rows * n_scenarios)
        base_normals_flat = base_normals.reshape(6, -1)
        z_corr_comp_flat = L @ base_normals_flat
        z_corr_comp = z_corr_comp_flat.reshape(6, n_rows, n_scenarios)
        
        # The sum of these correlated components has variance: 1^T R 1
        sum_corr_std = np.sqrt(np.sum(R))
        
        # Standardize the correlated sum to have Unit Variance
        # Shape: (n_rows, n_scenarios)
        z_corr_unit = np.sum(z_corr_comp, axis=0) / sum_corr_std
        
        # 2. Apply the correlated unit normal to the player's total score mean/std
        # Shape: (n_rows, n_scenarios)
        scenarios = means[:, np.newaxis] + z_corr_unit * stds[:, np.newaxis]
        
        # Map back to tensor
        for i, p_id in enumerate(p_ids):
            p_row_idx = player_idx_map[p_id]
            tensor[p_row_idx, t_idx, :] = scenarios[i]
            
    return tensor

def compute_squad_cvar(
    scenario_tensor: np.ndarray,
    squad_indices: list, # Indices in the player dimension of the tensor
    captain_idx_in_squad: int,
    alpha: float = 0.10,
) -> float:
    """
    Computes Conditional Value at Risk (CVaR) for a given squad.
    CVaR_alpha is the average of the worst alpha% of scenarios.
    """
    # Sum across selected players (for all gameweeks and scenarios)
    # Shape of squad_scenarios: (n_gw, n_scenarios)
    squad_scenarios = np.sum(scenario_tensor[squad_indices, :, :], axis=0)
    
    # Total points across the horizon per scenario
    # Shape: (n_scenarios,)
    total_points_per_scenario = np.sum(squad_scenarios, axis=0)
    
    # Add captain bonus (assuming captain is in squad)
    # We need the scenario-specific points for the captain
    cap_id_in_tensor = squad_indices[captain_idx_in_squad]
    cap_bonus = np.sum(scenario_tensor[cap_id_in_tensor, :, :], axis=0)
    total_points_per_scenario += cap_bonus
    
    # Find the alpha-quantile (Value at Risk)
    var_threshold = np.quantile(total_points_per_scenario, alpha)
    
    # CVaR is the average of all scenarios below VaR
    tail_scenarios = total_points_per_scenario[total_points_per_scenario <= var_threshold]
    
    return np.mean(tail_scenarios) if len(tail_scenarios) > 0 else var_threshold
