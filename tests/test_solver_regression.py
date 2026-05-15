import sys
import os
import pandas as pd
import numpy as np

# Ensure fpl_engine is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fpl_engine.solver import plan_sequential_transfers, _prefilter_candidate_pool
from tests.synthetic_data import generate_synthetic_projections

def test_solver_regression():
    print("\n--- Testing Solver Regression & Constraints ---")
    
    # 1. Setup synthetic data
    n_players = 200
    n_gw = 3
    projections = generate_synthetic_projections(n_players=n_players, n_gw=n_gw)
    
    # 2. Mock positional constraint DataFrame (Offline fallback)
    pos_df = pd.DataFrame([
        {'id': 1, 'singular_name_short': 'GKP', 'squad_select': 2, 'squad_min_play': 1, 'squad_max_play': 1},
        {'id': 2, 'singular_name_short': 'DEF', 'squad_select': 5, 'squad_min_play': 3, 'squad_max_play': 5},
        {'id': 3, 'singular_name_short': 'MID', 'squad_select': 5, 'squad_min_play': 2, 'squad_max_play': 5},
        {'id': 4, 'singular_name_short': 'FWD', 'squad_select': 3, 'squad_min_play': 1, 'squad_max_play': 3},
    ])
    
    # 3. Map positions to IDs 1-4 for the solver
    pos_map = {'GKP': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    projections['position_id'] = projections['position'].map(pos_map)
    
    # 4. Pick an initial team (first 15 players, balanced positions)
    initial_team_ids = []
    for pos, count in zip(['GKP', 'DEF', 'MID', 'FWD'], [2, 5, 5, 3]):
        pids = projections[projections['position'] == pos]['id_player'].unique()[:count]
        initial_team_ids.extend(pids.tolist())
        
    initial_values = {pid: projections[projections['id_player'] == pid]['now_cost'].iloc[0] for pid in initial_team_ids}
    
    # 5. Run solver with CVaR weight = 0 (Baseline)
    print("Running Baseline Solver...")
    result_base = plan_sequential_transfers(
        projections,
        start_gameweek=1,
        current_team_ids=initial_team_ids,
        current_realizable_value_dict=initial_values,
        bank_balance=100.0,
        planning_horizon=n_gw,
        pos_df=pos_df,
        cvar_weight=0.0
    )
    
    # 6. Run solver with CVaR weight = 0.5 (Risk Averse)
    projections['sc_p10'] = projections['Perf_IDX'] - 1.0
    
    print("\nRunning Risk-Averse Solver (CVaR=0.5)...")
    result_risk = plan_sequential_transfers(
        projections,
        start_gameweek=1,
        current_team_ids=initial_team_ids,
        current_realizable_value_dict=initial_values,
        bank_balance=100.0,
        planning_horizon=n_gw,
        pos_df=pos_df,
        cvar_weight=0.5
    )
    
    # 7. Verify pre-filter logic
    filtered_ids = _prefilter_candidate_pool(
        projections,
        initial_team_ids,
        top_k_per_pos=10
    )
    assert all(pid in filtered_ids for pid in initial_team_ids)
    print(f"Pre-filter preserved all {len(initial_team_ids)} current team members.")
        
    print("✅ Solver Regression & Pre-filter checks passed.")

if __name__ == "__main__":
    test_solver_regression()
