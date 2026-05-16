import pandas as pd
import numpy as np
import pytest
from fpl_engine.features import compute_coverage_flags

# Dummy parameters for testing
PARAMS = {
    'coverage_spike_z_gkp': 1.2,
    'coverage_spike_z_def': 1.5,
    'coverage_spike_z_mid': 1.7,
    'coverage_spike_z_fwd': 1.5,
    'coverage_min_spike_gkp': 60.0,
    'coverage_min_spike_def': 55.0,
    'coverage_min_spike_mid': 50.0,
    'coverage_min_spike_fwd': 55.0,
    'minutes_coverage_revert_w': 0.70,
}

@pytest.fixture
def mock_player_df():
    return pd.DataFrame([
        {'id': 1, 'team': 1, 'position': 'GKP'}, # True starter
        {'id': 2, 'team': 1, 'position': 'DEF'}, # True enabler
        {'id': 3, 'team': 1, 'position': 'MID'}, # True starter for MID
        {'id': 4, 'team': 1, 'position': 'MID'}, # Backup MID
        {'id': 8, 'team': 1, 'position': 'FWD'}, # True starter FWD
        {'id': 5, 'team': 1, 'position': 'FWD'}, # Backup FWD
        {'id': 6, 'team': 2, 'position': 'DEF'}, # Sustained multi-week starter (backup turned starter)
        {'id': 7, 'team': 2, 'position': 'DEF'}, # Previous starter for team 2 who got injured
    ])

@pytest.fixture
def mock_history_df():
    data = []
    # Create 25 fixtures of history
    for fx in range(1, 26):
        # Player 1 (True Starter GKP): plays 90 mins every game
        data.append({'id_player': 1, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': 90})
        
        # Player 2 (True Enabler DEF): plays ~60 mins every game
        data.append({'id_player': 2, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': 60})
        
        # Player 3 (True Starter MID): plays 90 mins every game
        data.append({'id_player': 3, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': 90})
        
        # Player 4 (Backup MID): 0 mins for 22 games, 90 mins for last 3 games
        mins = 90 if fx > 22 else 0
        data.append({'id_player': 4, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': mins})
        
        # Player 8 (True Starter FWD): plays 90 mins every game
        data.append({'id_player': 8, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': 90})
        
        # Player 5 (Backup FWD): 0 mins for 22 games, 90 mins for last 3 games
        data.append({'id_player': 5, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': mins})
        
        # Player 6 (Sustained multi-week starter DEF): 0 mins for 10 games, 90 mins for 15 games
        mins_sustained = 90 if fx > 10 else 0
        data.append({'id_player': 6, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': mins_sustained})

        # Player 7 (Previous starter DEF): 90 mins for 10 games, 0 mins for 15 games
        mins_prev = 90 if fx <= 10 else 0
        data.append({'id_player': 7, 'id_fixture': fx, 'kickoff_time': fx, 'minutes': mins_prev})
        
    return pd.DataFrame(data)

@pytest.fixture
def mock_fixture_player_df():
    # Only need 1 row per player representing the current state for future fixtures
    return pd.DataFrame([
        {'id_player': 1, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 90.0},
        {'id_player': 2, 'recent_minutes_form': 60.0, 'finished': False, 'minutes_per_game': 60.0},
        {'id_player': 3, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 90.0},
        {'id_player': 4, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 15.0},
        {'id_player': 8, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 90.0},
        {'id_player': 5, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 15.0},
        {'id_player': 6, 'recent_minutes_form': 90.0, 'finished': False, 'minutes_per_game': 60.0},
        {'id_player': 7, 'recent_minutes_form': 0.0,  'finished': False, 'minutes_per_game': 30.0},
    ])

def test_compute_coverage_flags(mock_fixture_player_df, mock_history_df, mock_player_df):
    result = compute_coverage_flags(
        mock_fixture_player_df, 
        mock_history_df, 
        mock_player_df, 
        PARAMS, 
        lookback_window=20
    )
    
    # Assert Player 1 (True Starter) - Should not be suppressed
    p1 = result[result['id_player'] == 1].iloc[0]
    assert p1['depth_rank'] == 1
    assert p1['season_p50'] == 90.0
    assert p1['is_coverage_spike'] == False

    # Assert Player 2 (True Enabler) - Should not be suppressed
    p2 = result[result['id_player'] == 2].iloc[0]
    assert p2['depth_rank'] == 1
    assert p2['season_p50'] == 60.0
    assert p2['is_coverage_spike'] == False
    
    # Assert Player 4 (Backup MID) - Should be flagged as spike
    p4 = result[result['id_player'] == 4].iloc[0]
    assert p4['depth_rank'] >= 2  # Player 3 has rank 1
    assert p4['season_p50'] == 0.0 # From the 20 matches prior to the last 3
    assert p4['is_coverage_spike'] == True
    
    # Assert Player 5 (Backup FWD) - Should be flagged as spike
    p5 = result[result['id_player'] == 5].iloc[0]
    assert p5['depth_rank'] >= 2 
    assert p5['season_p50'] == 0.0 
    assert p5['is_coverage_spike'] == True

    # Assert Player 8 (True Starter FWD) - Should not be suppressed
    p8 = result[result['id_player'] == 8].iloc[0]
    assert p8['depth_rank'] == 1
    assert p8['season_p50'] == 90.0
    assert p8['is_coverage_spike'] == False

    # Assert Player 6 (Sustained multi-week starter) - Should NOT be suppressed
    p6 = result[result['id_player'] == 6].iloc[0]
    # In the prior 20 matches (fx 3 to 22), Player 6 played 0 mins in 8 games, 90 in 12 games. 
    # Median of [0]*8 + [90]*12 is 90.0
    assert p6['season_p50'] == 90.0
    assert p6['is_coverage_spike'] == False

def test_scoring_suppression(mock_fixture_player_df, mock_history_df, mock_player_df):
    flags_df = compute_coverage_flags(
        mock_fixture_player_df, 
        mock_history_df, 
        mock_player_df, 
        PARAMS, 
        lookback_window=20
    )
    
    # Mocking the DataFrame as it enters B8.5 in scoring.py
    df = flags_df.copy()
    df['minutes_IDX'] = df['recent_minutes_form'] # Mock pre-suppression minutes
    
    # Run B8.5 logic
    COVERAGE_REVERT_W = PARAMS.get('minutes_coverage_revert_w', 0.70)
    coverage_mask = (
        df['is_coverage_spike'].fillna(False).astype(bool)
        & (df['depth_rank'].fillna(99).astype(int) >= 2)
    )

    if coverage_mask.any():
        df.loc[coverage_mask, 'minutes_IDX'] = (
            (1.0 - COVERAGE_REVERT_W) * df.loc[coverage_mask, 'minutes_IDX']
            + COVERAGE_REVERT_W       * df.loc[coverage_mask, 'season_p50'].fillna(0)
        )

    # Player 1 (True Starter) should remain 90
    p1 = df[df['id_player'] == 1].iloc[0]
    assert p1['minutes_IDX'] == 90.0
    
    # Player 4 (Backup MID) should be suppressed
    # minutes_IDX = (1 - 0.7)*90 + 0.7*0 = 27
    p4 = df[df['id_player'] == 4].iloc[0]
    assert abs(p4['minutes_IDX'] - 27.0) < 1e-5
    
    # Player 6 (Sustained starter) should remain 90
    p6 = df[df['id_player'] == 6].iloc[0]
    assert p6['minutes_IDX'] == 90.0
