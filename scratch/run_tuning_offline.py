import sys
import os
import asyncio
import pandas as pd
import numpy as np
import json
from unittest.mock import MagicMock, patch

# Add workspace to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import fpl_engine.data as data
from fpl_engine.tuning import auto_tune_if_needed

# --- MOCKING DATA FUNCTIONS DIRECTLY ---
def mock_get_current_players_df():
    hist = pd.read_parquet('raw_history_cache.parquet')
    player_ids = hist['id_player'].unique()
    
    df = pd.DataFrame({'id': player_ids, 'id_player': player_ids})
    df['now_cost'] = 55
    df['selected_by_percent'] = 10.0
    df['team'] = (df['id'] % 20) + 1 
    df['web_name'] = df['id'].astype(str)
    df['position'] = 'MID'
    df['team_name'] = 'Team'
    df['minutes'] = 1000
    df['total_points'] = 100
    
    # Required columns for tuning filters and features
    cols = ['form', 'starts_per_90', 'chance_of_playing_this_round', 
                'chance_of_playing_next_round', 'expected_goals_per_90', 
                'expected_assists_per_90', 'expected_goal_involvements_per_90', 
                'expected_goals_conceded_per_90', 'goals_conceded_per_90', 
                'clean_sheets_per_90', 'defensive_contribution_per_90',
                'influence_per_90', 'creativity_per_90', 'threat_per_90',
                'ict_index_per_90', 'yellow_cards_per_90', 'red_cards_index_per_90',
                'saves_per_90', 'bonus_per_90', 'bps_per_90',
                'goals_scored', 'goals_conceded', 'expected_goals_conceded', 'expected_goals']
    for col in cols:
        df[col] = 1.0
    
    df['game_played'] = 20
    df['points_per_game'] = 5.0
    df['starts'] = 18
    df['start_per_gameplayed'] = 0.9
    df['start_share_total_game'] = 0.5
    
    return df

def mock_get_team_df():
    return pd.DataFrame({
        'id': range(1, 21),
        'name': [f'Team {i}' for i in range(1, 21)],
        'short_name': [f'T{i}' for i in range(1, 21)],
        'strength_overall_home': 1200, 'strength_overall_away': 1200,
        'strength_attack_home': 1200, 'strength_attack_away': 1200,
        'strength_defence_home': 1200, 'strength_defence_away': 1200
    })

def mock_get_fixture_df():
    hist = pd.read_parquet('raw_history_cache.parquet')
    hist['team'] = (hist['id_player'] % 20) + 1
    
    f1 = hist[['id_fixture', 'gameweek', 'kickoff_time', 'team', 'opponent', 'was_home']].copy()
    f1['finished'] = True
    f1['team_difficulty'] = 3
    f1['opp_difficulty'] = 3
    f1['team_h'] = np.where(f1['was_home'], f1['team'], f1['opponent'])
    f1['team_a'] = np.where(f1['was_home'], f1['opponent'], f1['team'])
    
    f2 = pd.DataFrame({
        'id_fixture': [9999], 'gameweek': [37], 'kickoff_time': ['2026-05-20T12:00:00Z'],
        'team': [1], 'opponent': [2], 'was_home': [True],
        'finished': [False], 'team_difficulty': [3], 'opp_difficulty': [3],
        'team_h': [1], 'team_a': [2]
    })
    
    return pd.concat([f1, f2]).drop_duplicates(['id_fixture', 'team'])

async def mock_fetch_raw_history_cache(*args, **kwargs):
    df = pd.read_parquet('raw_history_cache.parquet')
    df['team'] = (df['id_player'] % 20) + 1
    df['team_h'] = np.where(df['was_home'], df['team'], df['opponent'])
    df['team_a'] = np.where(df['was_home'], df['opponent'], df['team'])
    return df

async def main():
    print("Starting Offline Tuning Runner (Complete Column Mocks)...")
    
    with patch('fpl_engine.data.get_current_players_df', side_effect=mock_get_current_players_df), \
         patch('fpl_engine.data.get_team_df', side_effect=mock_get_team_df), \
         patch('fpl_engine.data.get_fixture_df', side_effect=mock_get_fixture_df), \
         patch('fpl_engine.data.fetch_raw_history_cache', side_effect=mock_fetch_raw_history_cache), \
         patch('fpl_engine.data.get_max_finished_gameweek', return_value=36):
        
        await auto_tune_if_needed(current_gw=36, force=True, n_trials_override=2)

if __name__ == "__main__":
    asyncio.run(main())
