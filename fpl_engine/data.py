import pandas as pd
import numpy as np
import requests
import json
import asyncio
import concurrent.futures
import os
import time
from datetime import datetime
from scipy.optimize import minimize
from tqdm.auto import tqdm

def get_fpl_gameweek_summary():
    """
    Fetches the bootstrap-static data from the FPL API and extracts gameweek (event) summaries.

    Returns:
        pd.DataFrame: A pandas DataFrame containing details for each gameweek, or None if an error occurs.
    """
    api_url = "https://fantasy.premierleague.com/api/bootstrap-static/"

    try:
        # Make a GET request to the FPL API
        response = requests.get(api_url)
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

        # Parse the JSON response
        data = response.json()

        # The gameweek information is under the 'events' key
        events = data.get('events', [])

        if not events:
            print("No event data found in the FPL API response.")
            return None

        # Prepare a list of dictionaries to easily convert to a DataFrame
        gameweek_data_list = []
        for event in events:
            gameweek_id = event.get('id')
            name = event.get('name')
            finished = event.get('finished')
            is_current = event.get('is_current')
            is_next = event.get('is_next')
            avg_points = event.get('average_entry_score')
            highest_score = event.get('highest_score')
            deadline_time = event.get('deadline_time')
            chip_plays = event.get('chip_plays')

            gameweek_data_list.append({
                "Gameweek ID": gameweek_id,
                "Name": name,
                "Finished": finished,
                "Is Current": is_current,
                "Is Next": is_next,
                "Average Score": avg_points,
                "Highest Score": highest_score,
                "Deadline Time": deadline_time,
                "Chip Plays": json.dumps(chip_plays) # Store chip plays as a JSON string for DataFrame compatibility
            })

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(gameweek_data_list)
        return df

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from FPL API: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# --- CELL 8 ---
def get_fpl_gameweek_data(fpl_id):
    """Fetches FPL history data and processes gameweek information.

    Args:
        fpl_id: The Fantasy Premier League ID.

    Returns:
        A pandas DataFrame containing processed gameweek data.
    """
    # Fetch static data (events)
    url_static = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response_static = requests.get(url_static)
    response_static.raise_for_status()
    data_static = response_static.json()

    events_df = pd.DataFrame(data_static['events'])

    def extract_chip_plays(chip_list):
        chips = {}
        if isinstance(chip_list, list):
            for chip_info in chip_list:
                chips[chip_info['chip_name'] + "_played"] = chip_info['num_played']
        return chips

    chip_plays_df = events_df['chip_plays'].apply(extract_chip_plays).apply(pd.Series)
    events_df = pd.concat([events_df, chip_plays_df], axis=1)

    # Fetch user's history data
    url_history = f"https://fantasy.premierleague.com/api/entry/{fpl_id}/history"
    response_history = requests.get(url_history)
    response_history.raise_for_status()
    data_history = response_history.json()

    my_performance_df = pd.DataFrame(data_history['current'])

    # Check if 'chips' list is not empty before creating DataFrame
    if data_history.get('chips') and isinstance(data_history['chips'], list) and data_history['chips']:
        my_chips_df = pd.DataFrame(data_history['chips'])[['event', 'name']].rename(columns={'name': 'used_chip_name'})
    else:
        # Create an empty DataFrame with the expected columns if no chips are used
        my_chips_df = pd.DataFrame(columns=['event', 'used_chip_name'])


    my_performance_df = pd.merge(my_performance_df, my_chips_df, left_on='event', right_on='event', how='left')
    my_performance_df.columns = ["my_" + col for col in my_performance_df.columns]


    # Join dataframes on event/my_event
    fpl_gameweeks = pd.merge(events_df, my_performance_df, left_on='id', right_on='my_event', how='left')

    # Set relevant columns to NaN if gameweek is not finished
    unfinished_mask = fpl_gameweeks['finished'] == False
    fpl_gameweeks.loc[unfinished_mask, ['average_entry_score', 'highest_score', 'ranked_count','my_points']] = pd.NA


    fpl_gameweeks['my_overall_percentile_rank'] = round(1 - (fpl_gameweeks['my_overall_rank'] / fpl_gameweeks['ranked_count']),3)*100
    fpl_gameweeks['my_week_percentile_rank'] = round(1 - (fpl_gameweeks['my_rank'] / fpl_gameweeks['ranked_count']),3)*100

    return fpl_gameweeks[[
        'id', 'name', 'deadline_time', 'average_entry_score',
       'finished', 'data_checked', 'highest_score',
       'is_previous', 'is_current', 'is_next',
       'ranked_count', 'most_selected',
       'transfers_made', 'most_captained', 'most_vice_captained',
       'bboost_played', '3xc_played', 'freehit_played', 'wildcard_played',
       'my_total_points', 'my_rank',
       'my_overall_rank','my_overall_percentile_rank','my_points', 'my_week_percentile_rank', 'my_bank', 'my_value',
       'my_event_transfers', 'my_event_transfers_cost', 'my_points_on_bench','my_used_chip_name'
       ]].rename(columns={
        'id': 'gameweek_id',
        'my_points': 'my_week_points',})

# --- CELL 9 ---
def get_max_finished_gameweek():
    """
    Gets the maximum finished gameweek number from the gameweek summary DataFrame.

    Args:
        gameweek_summary_df (pd.DataFrame): DataFrame containing gameweek summaries.

    Returns:
        int or None: The maximum finished gameweek ID as an integer, or None if not found.
    """
    gameweek_summary_df = get_fpl_gameweek_summary()
    if gameweek_summary_df is None:
        print("Failed to fetch gameweek summary data.")
        return None
    finished_gameweeks_rows = gameweek_summary_df[gameweek_summary_df['Finished'] == True]
    if not finished_gameweeks_rows.empty:
        # Ensure the 'Gameweek ID' is treated as an integer
        return int(finished_gameweeks_rows['Gameweek ID'].max())
    else:
        print("No finished gameweeks found in gameweek_summary.")
        return None

# --- CELL 10 ---
def get_current_gameweek():
    """
    Gets the maximum finished gameweek number from the gameweek summary DataFrame.

    Args:
        gameweek_summary_df (pd.DataFrame): DataFrame containing gameweek summaries.

    Returns:
        int or None: The maximum finished gameweek ID as an integer, or None if not found.
    """
    gameweek_summary_df = get_fpl_gameweek_summary()
    if gameweek_summary_df is None:
        print("Failed to fetch gameweek summary data.")
        return None
    current_gameweeks_rows = gameweek_summary_df[gameweek_summary_df['Is Current'] == True]
    if not current_gameweeks_rows.empty:
        # Ensure the 'Gameweek ID' is treated as an integer
        return int(current_gameweeks_rows['Gameweek ID'].iloc[0])
    else:
        print("No finished gameweeks found in gameweek_summary.")
        return None

# --- CELL 11 ---
def get_my_player_ids(manager_id, gameweek=1):
    """
    Fetches the list of player IDs for a specific FPL manager's team.
    This is used to get your initial squad.
    """
    try:
        url = f"https://fantasy.premierleague.com/api/entry/{manager_id}/event/{gameweek}/picks/"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data['active_chip'] == 'freehit':
            if gameweek <= 1:
                print("Free hit used in GW1. Fetching current GW team IDs instead.")
                return [player['element'] for player in data['picks']]

            print(f"Free hit was used in previous week, fetching your team IDs from Gameweek {gameweek-1} instead")
            return get_my_player_ids(manager_id, gameweek-1)
        else:
          player_ids = [player['element'] for player in data['picks']]
          print(f"Successfully fetched your team IDs for Gameweek {gameweek}:")
          print(player_ids)
          return player_ids
    except requests.exceptions.RequestException as e:
        print(f"Error fetching your team data: {e}")
        print("Please check if your FPL Manager ID is correct.")
        return []

# --- CELL 12 ---
def get_current_players_df():
    """
    Fetches current player data and adds average minutes from the last 2 finished Gameweeks
    using the optimized 'live' endpoint to minimize API calls.
    """
    # 1. Fetch Base Data (Bootstrap Static)
    base_url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()

    # Create base dataframes
    elements_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    pos_df = pd.DataFrame(data['element_types'])
    events_df = pd.DataFrame(data['events'])

    # ------------------------------------------------------------------

    # Standard Transformations (Your original logic)
    if 'now_cost' in elements_df.columns:
      elements_df['now_cost'] = elements_df['now_cost'] / 10.0

    pos_map = pos_df.set_index('id')['singular_name_short']
    team_map = teams_df.set_index('id')['name']
    elements_df['position'] = elements_df['element_type'].map(pos_map)
    elements_df['team_name'] = elements_df['team'].map(team_map)

    # Handle numeric columns safely
    numeric_cols = ['total_points', 'points_per_game', 'starts', 'minutes',
                'influence', 'creativity', 'threat', 'ict_index',
                'yellow_cards', 'red_cards', 'saves', 'goals_scored',
                'expected_goals', 'expected_assists', 'expected_goal_involvements',
                'expected_goals_conceded', 'goals_conceded', 'clean_sheets']

    # Ensure columns exist and fill NaNs before division
    for col in numeric_cols:
        if col not in elements_df.columns:
            elements_df[col] = 0
        elements_df[col] = pd.to_numeric(elements_df[col], errors='coerce').fillna(0)

    # Metric Calculations
    elements_df['game_played'] = round(elements_df['total_points'] / (elements_df['points_per_game'] + 1e-10),0)
    elements_df['start_per_gameplayed'] = (elements_df['starts'] / (elements_df['game_played'] + 1e-10))
    elements_df['start_share_total_game'] = (elements_df['starts'] / (get_current_gameweek() + 1e-10))
    elements_df['minutes_per_game'] = (elements_df['minutes'] / (elements_df['game_played'] + 1e-10)).clip(lower=0, upper=90)

    elements_df['influence_per_90'] = elements_df['influence'] / (elements_df['minutes'] + 1e-10) * 90
    elements_df['creativity_per_90'] = elements_df['creativity'] / (elements_df['minutes'] + 1e-10 ) * 90
    elements_df['threat_per_90'] = elements_df['threat'] / (elements_df['minutes'] + 1e-10) * 90
    elements_df['ict_index_per_90'] = elements_df['ict_index'] / (elements_df['minutes'] + 1e-10) * 90

    elements_df['bps_per_90'] = elements_df['bps'] / (elements_df['minutes'] + 1e-10) * 90
    elements_df['bonus_per_90'] = elements_df['bonus'] / (elements_df['minutes'] + 1e-10) * 90

    elements_df['yellow_cards_per_90'] = elements_df['yellow_cards'] / (elements_df['minutes'] + 1e-10) * 90
    elements_df['red_cards_index_per_90'] = elements_df['red_cards'] / (elements_df['minutes'] + 1e-10) * 90
    elements_df['saves_per_90'] = elements_df['saves'] / (elements_df['minutes'] + 1e-10) * 90

    # Calculate global means across the whole dataset (excluding players with 0 mins)
    active_players = elements_df[elements_df['minutes'] > 0]
    mean_goals = (active_players['goals_scored'].sum() / (active_players['minutes'].sum() + 1e-10)) * 90
    mean_xg = (active_players['expected_goals'].sum() / (active_players['minutes'].sum() + 1e-10)) * 90
    mean_GC = (active_players['goals_conceded'].sum() / (active_players['minutes'].sum() + 1e-10)) * 90
    mean_xGC = (active_players['expected_goals_conceded'].sum() / (active_players['minutes'].sum() + 1e-10)) * 90

    # C is your "confidence" in the prior. Events (Goals/xG) are rare. You need a higher threshold to avoid "luck" bias.
    # Add as parameter or global constant with comment
    C_FINISHING  = 20     # ~20–25 most common for goals/xG in PL
    C_PROTECTIVE = 30     # slightly higher for GK/defenders (goals conceded are even noisier)
                          # because defenders/GKs face more events but variance remains high

    # Then use separately if desired:
    elements_df['finishing_factor'] = (
        (elements_df['goals_scored']   + C_FINISHING  * mean_goals) /
        (elements_df['expected_goals'] + C_FINISHING  * mean_xg)
    )

    elements_df['protective_factor'] = (
        (elements_df['goals_conceded']         + C_PROTECTIVE * mean_GC) /
        (elements_df['expected_goals_conceded'] + C_PROTECTIVE * mean_xGC)
    )

    elements_df['total_non_minutes_points'] = elements_df['total_points'] - (elements_df['game_played'] * ((elements_df['minutes_per_game'] > 0).astype(int) + (elements_df['minutes_per_game'] >= 60).astype(int)))

    # Added necessary columns for calculation to the return list if they were missing
    return elements_df[[
            'id', 'now_cost', 'selected_by_percent', 'team', 'web_name',
            'position', 'team_name', 'game_played', 'total_points',
            'points_per_game','form',
            'starts_per_90', 'starts',
            'start_per_gameplayed' ,
            'start_share_total_game',
            'chance_of_playing_this_round',
            'chance_of_playing_next_round', 'minutes_per_game',
            'minutes',
            'finishing_factor',
            'protective_factor',
            'expected_goals',
            'goals_scored',
            'expected_assists',
            'assists',
            'goals_conceded',
            'expected_goals_conceded',
            'clean_sheets',
            'total_non_minutes_points',
            'ict_index',
            'creativity',
            'threat',
            'bps_per_90',
            'bonus_per_90',
            'yellow_cards_per_90', 'red_cards_index_per_90', 'saves_per_90',
            'influence_per_90', 'creativity_per_90', 'threat_per_90',
            'ict_index_per_90', 'expected_goals_per_90',
            'expected_assists_per_90', 'expected_goal_involvements_per_90',
            'expected_goals_conceded_per_90', 'goals_conceded_per_90',
            'clean_sheets_per_90','defensive_contribution_per_90'
    ]]

# --- CELL 13 ---
def get_pos_constraint_df():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    pos_df = pd.DataFrame(data['element_types'])

    return pos_df[
        ['id',
         'singular_name_short',
         'squad_select',
         'squad_min_play',
         'squad_max_play',
       ]
        ]

# --- CELL 14 ---
def get_team_df():
    url = "https://fantasy.premierleague.com/api/bootstrap-static/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    team_df = pd.DataFrame(data['element_types'])

    return pd.DataFrame(data['teams'])[['code', 'id', 'name',
       'position', 'short_name', 'strength','strength_overall_home', 'strength_overall_away',
       'strength_attack_home', 'strength_attack_away', 'strength_defence_home',
       'strength_defence_away', 'pulse_id']]

# --- CELL 15 ---
def get_fixture_df():
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    # 1. Load the initial data (Added scores and finished status)
    df = pd.DataFrame(data)[['event', 'id', 'kickoff_time', 'team_a', 'team_h',
                             'team_a_score', 'team_h_score', 'finished', 'stats']]

    # 2. Create the "Home" perspective DataFrame
    df_home = df.rename(columns={
        'team_h': 'team',
        'team_a': 'opponent',
        'team_h_score': 'team_score',
        'team_a_score': 'opponent_score'
    })
    df_home['is_home'] = True

    # 3. Create the "Away" perspective DataFrame
    df_away = df.rename(columns={
        'team_a': 'team',
        'team_h': 'opponent',
        'team_a_score': 'team_score',
        'team_h_score': 'opponent_score'
    })
    df_away['is_home'] = False

    # 4. Concatenate them together and sort
    final_df = pd.concat([df_home, df_away])
    final_df = final_df.sort_values(by=['id', 'kickoff_time'])
    final_df = final_df.rename(columns={'event': 'gameweek'})

    # 5. Return requested columns (Now includes contextual match outcomes)
    return final_df[['gameweek', 'id', 'kickoff_time', 'team', 'opponent',
                     'is_home', 'team_score', 'opponent_score', 'finished', 'stats']].reset_index(drop=True)

# --- CELL 16 ---
def get_dynamic_weights(
    df_gameweek,
    current_gw,
    max_diff_weight:   float = 0.13,
    max_upside_weight: float = 0.12,
    target_low:        float = 88,   # 88 = top 12%
    target_high:       float = 95,  # 95 = top 5%
) -> dict:
    """
    Weight logic anchored to a specific rank TARGET ZONE.

    ┌──────────────────┬────────────────────────────────────────────────┐
    │ Rank Percentile  │ Mode & Rationale                               │
    ├──────────────────┼────────────────────────────────────────────────┤
    │ 0  – 60          │ CHASE HARD  — far from target, take risks      │
    │ 60 – target_low  │ PUSH        — closing in, still need gains     │
    │ target_low – high│ PROTECT+    — inside zone, small edges only    │
    │ target_high+     │ LOCK DOWN   — above target, preserve lead      │
    └──────────────────┴────────────────────────────────────────────────┘
    """

    # --- 1. Extract rank ---
    rank_pct = None
    source = None
    try:
        row = df_gameweek.loc[df_gameweek['gameweek_id'] == current_gw]
        if not row.empty:
            rank_pct = float(row['my_overall_percentile_rank'].values[0])
            source = 'live'
    except (KeyError, IndexError, TypeError):
        pass

    if rank_pct is None:
        rank_pct = 50.0
        source = 'default (data missing)'

    rank_pct = float(np.clip(rank_pct, 0, 100))

    # --- 2. Target-anchored decay ---
    # Sigmoid centred at target_low (where you start protecting)
    # Below target_low: decay > 1 (aggressive)
    # Above target_high: decay at floor (conservative)

    # Primary sigmoid — transitions around your target entry point
    sigmoid_primary = 2.0 / (1 + np.exp(0.10 * (rank_pct - target_low)))

    # Secondary dampener — extra suppression once above target_high
    above_target = max(0, rank_pct - target_high)
    sigmoid_secondary = np.exp(-0.08 * above_target)

    decay_factor = float(np.clip(
        sigmoid_primary * sigmoid_secondary,
        0.25,   # floor: always keep some differential exposure
        1.60    # ceiling: max aggression when far from target
    ))

    # --- 3. Compute weights ---
    # diff decays slower — low-owned quality players are lower risk
    # upside decays faster — captain swing is binary win/lose
    diff_weight   = float(np.clip(
        max_diff_weight   * decay_factor,
        max_diff_weight   * 0.25,   # floor at 25% of max
        max_diff_weight   * 1.60    # ceiling at 160% of max
    ))
    upside_weight = float(np.clip(
        max_upside_weight * decay_factor * 0.85,  # 15% faster decay than diff
        max_upside_weight * 0.20,   # higher floor — always some captain lean
        max_upside_weight * 1.60
    ))

    # --- 4. Mode + proximity to target ---
    gap_to_target = target_low - rank_pct   # positive = below target

    if rank_pct <= 60:
        mode = 'CHASE HARD 🔴'
    elif rank_pct < target_low:
        mode = f'PUSH 🟠  ({gap_to_target:.1f}% from target)'
    elif rank_pct <= target_high:
        mode = f'PROTECT+ 🟡  (inside target zone)'
    else:
        mode = f'LOCK DOWN 🟢  ({rank_pct - target_high:.1f}% above target)'

    return {
        'diff_weight':    round(diff_weight, 4),
        'upside_weight':  round(upside_weight, 4),
        'decay_factor':   round(decay_factor, 4),
        'rank_pct':       rank_pct,
        'gap_to_target':  round(gap_to_target, 1),
        'mode':           mode,
        'source':         source,
    }

# --- CELL 17 ---

def _fetch_player_sync(p_id):
    """Synchronous fetch for a single player's match history.
    Runs inside a ThreadPoolExecutor to achieve concurrency without aiohttp,
    which is incompatible with Python 3.14's asyncio.current_task() changes.
    """
    url = f"https://fantasy.premierleague.com/api/element-summary/{p_id}/"
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return []
        data = response.json()
    except Exception:
        return []

    player_records = []
    # Removed the 'if match['minutes'] > 0:' filter to capture all matches
    for match in data.get('history', []):
        player_records.append({
            'id_player': p_id,
            'id_fixture': match['fixture'],
            'gameweek': match['round'],
            'kickoff_time': match['kickoff_time'],
            'opponent': match['opponent_team'],
            'was_home': match['was_home'],
            'minutes': match['minutes'],
            'expected_goals': float(match['expected_goals']),
            'expected_assists': float(match['expected_assists']),
            'expected_goals_conceded': float(match['expected_goals_conceded']),
            'threat': float(match['threat']),
            'creativity': float(match['creativity']),
            'goals_scored': match['goals_scored'],
            'goals_conceded': match['goals_conceded'],
            'saves': match['saves'],
            'clean_sheets': match['clean_sheets'],
            'bonus':match['bonus'],
            'bps':match['bps'],
            'defensive_contribution':match['defensive_contribution'],
            'actual_points': match['total_points'],
        })
    return player_records

async def _fetch_all_async(active_player_ids):
    """Fetch all player histories concurrently using a thread pool.
    Uses ThreadPoolExecutor + synchronous requests instead of aiohttp
    to avoid Python 3.14 asyncio.current_task() incompatibility.
    """
    loop = asyncio.get_running_loop()

    with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        futures = [
            loop.run_in_executor(executor, _fetch_player_sync, p_id)
            for p_id in active_player_ids
        ]

        results = []
        pbar = tqdm(total=len(futures), desc="Fetching Match History")
        for coro in asyncio.as_completed(futures):
            results.append(await coro)
            pbar.update(1)
        pbar.close()

    return [record for sublist in results for record in sublist]

async def fetch_raw_history_cache(active_player_ids, use_cache=True, cache_timeout_hours=12):
    """Fetch raw match history with caching support."""
    cache_file = "raw_history_cache.parquet"

    if use_cache and os.path.exists(cache_file):
        file_age_seconds = time.time() - os.path.getmtime(cache_file)
        if file_age_seconds < (cache_timeout_hours * 3600):
            print(f"Loading raw match history from {cache_file} (Age: {file_age_seconds/3600:.1f} hours)...")
            return pd.read_parquet(cache_file)
        else:
            print(f"Cache {cache_file} expired (older than {cache_timeout_hours} hours). Fetching fresh data...")

    records = await _fetch_all_async(active_player_ids)

    raw_df = pd.DataFrame(records)

    if use_cache:
        raw_df.to_parquet(cache_file, index=False)
        print(f"Saved {len(raw_df)} match records to {cache_file}.")

    return raw_df

# --- CELL 19 ---
def enforce_datatypes(df, numeric_threshold=1.0):
    """
    Attempts to convert columns to numeric if the percentage of values that
    can be converted without error meets or exceeds the numeric_threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numeric_threshold (float, optional): The percentage threshold (between 0.0 and 1.0)
                                             for automatic numeric conversion. Defaults to 1.0 (100%).

    Returns:
        pd.DataFrame: The DataFrame with columns converted to numeric where the
                      threshold is met.
    """
    df_cleaned = df.copy()

    # Attempt to convert columns to numeric based on the threshold
    for col in df_cleaned.columns:
        # Attempt conversion and count how many values are not NaN after coercion
        numeric_series = pd.to_numeric(df_cleaned[col], errors='coerce')
        non_nan_count = numeric_series.notna().sum()

        # Calculate the percentage of non-NaN values after attempted conversion
        convertible_percentage = non_nan_count / len(df_cleaned) if len(df_cleaned) > 0 else 0

        # If the percentage meets the threshold, convert the column
        if convertible_percentage >= numeric_threshold:
            df_cleaned[col] = numeric_series
        # Else, keep the original data type (no change needed)

    return df_cleaned

# --- CELL 20 ---
