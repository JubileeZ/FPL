import pandas as pd
import numpy as np
from typing import Tuple, Dict

def generate_synthetic_history(n_players: int = 50, n_fixtures: int = 10) -> pd.DataFrame:
    """
    Creates a mock player history DataFrame with controlled scoring patterns.
    Used to test the covariance estimator and GARCH volatility engine.
    """
    rows = []
    player_ids = np.arange(1, n_players + 1)
    
    for p_id in player_ids:
        # Assign a random "quality" level to the player
        quality = np.random.uniform(0.1, 0.9)
        
        for f_idx in range(n_fixtures):
            # Base minutes (randomly rotated or starter)
            is_starter = np.random.choice([0, 1], p=[0.2, 0.8])
            minutes = 90 if is_starter else np.random.choice([0, 15, 30])
            
            # Goals and assists linked to quality
            goals = np.random.poisson(0.3 * quality) if minutes > 60 else 0
            assists = np.random.poisson(0.2 * quality) if minutes > 60 else 0
            
            # Clean sheets (simplified)
            clean_sheets = np.random.choice([0, 1], p=[0.7, 0.3]) if minutes > 60 else 0
            
            # Bonus (correlated to goals and assists)
            bonus = 0
            if goals > 0 or assists > 0:
                bonus = np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
            
            rows.append({
                'id_player': p_id,
                'fixture_id': f_idx,
                'minutes': minutes,
                'goals_scored': goals,
                'assists': assists,
                'clean_sheets': clean_sheets,
                'goals_conceded': 0 if clean_sheets else np.random.randint(1, 4),
                'saves': np.random.randint(0, 5) if p_id % 10 == 0 else 0, # Only some "GKPs"
                'bonus': bonus,
                'bps': (goals * 24 + assists * 9 + clean_sheets * 12) + np.random.randint(0, 10)
            })
            
    return pd.DataFrame(rows)

def generate_synthetic_projections(n_players: int = 100, n_gw: int = 6) -> pd.DataFrame:
    """
    Generates mock projection data (Perf_IDX, score_std) for solver testing.
    """
    rows = []
    for p_id in range(1, n_players + 1):
        # Base cost and position
        cost = np.random.uniform(4.0, 12.0)
        pos = np.random.choice(['GKP', 'DEF', 'MID', 'FWD'])
        team = f"Team_{p_id % 20}"
        
        for gw in range(1, n_gw + 1):
            perf_idx = np.random.uniform(2.0, 8.0)
            # Higher score_std for attackers
            std_base = 2.0 if pos in ['MID', 'FWD'] else 1.5
            score_std = std_base + np.random.uniform(0.1, 1.0)
            
            rows.append({
                'id_player': p_id,
                'gameweek': gw,
                'web_name': f"Player_{p_id}",
                'position': pos,
                'team_name': team,
                'now_cost': cost,
                'selected_by_percent': np.random.uniform(1, 40),
                'Perf_IDX': perf_idx,
                'score_std': score_std,
                'ceiling_score': perf_idx + 1.5 * score_std,
                'custom_score': perf_idx + 0.1 * (perf_idx + 1.5 * score_std),
                'minutes_IDX': 80 if np.random.random() > 0.1 else 0,
                'GOAL_INDEX': 0.1, 'ASSIST_INDEX': 0.1, 'CLEAN_SHEET_INDEX': 0.3,
                'bonus_component': 0.5, 'defcon_component': 0.1,
                'actual_minutes': 90,
                'recent_minutes_form': 80, 'finishing_factor': 1.0, 'protective_factor': 1.0,
                'fixture_attack_multiplier': 1.0, 'fixture_defence_multiplier': 1.0,
                'fixture_calibrated_points': perf_idx,
                'start_per_gameplayed': 0.9, 'consecutive_start_streak': 5, 'hybrid_bps_abs': 20,
                'dynamic_upside': 1.5 * score_std,
                'raw_bonus_multiplier': 1.0
            })
            
    return pd.DataFrame(rows)

if __name__ == "__main__":
    history = generate_synthetic_history()
    print(f"Generated synthetic history with {len(history)} rows.")
    projections = generate_synthetic_projections()
    print(f"Generated synthetic projections with {len(projections)} rows.")
