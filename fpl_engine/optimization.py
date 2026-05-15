import pandas as pd
import numpy as np
import requests
import json
import asyncio
import aiohttp
import os
import pickle
import time
from datetime import datetime
from scipy.stats import poisson, norm
from scipy.optimize import minimize
import pulp
from optuna.samplers import GridSampler
import optuna
import optunahub
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
import warnings
from tqdm.auto import tqdm

def evaluate_fpl_duel_discrete(df, player_a_id, player_b_id, elo_win, elo_loss):
    """
    Calculates expected ELO using the Skellam distribution.
    Automatically derives the draw ELO impact.
    """
    # Calculate draw ELO internally
    elo_draw = (elo_win + elo_loss) / 2.0

    # Extract player data
    player_a = df[df['id_player'] == player_a_id].iloc[0]
    player_b = df[df['id_player'] == player_b_id].iloc[0]

    # Mean is Perf_IDX
    mu_a = player_a['Perf_IDX']
    mu_b = player_b['Perf_IDX']

    # Approximate Variance using the 95th percentile ceiling
    var_a = max((player_a['ceiling_score'] - mu_a) / 1.645, 1.0) ** 2
    var_b = max((player_b['ceiling_score'] - mu_b) / 1.645, 1.0) ** 2

    # Difference parameters
    mu_diff = mu_a - mu_b
    var_diff = var_a + var_b

    # Skellam parameters
    if var_diff <= abs(mu_diff):
        var_diff = abs(mu_diff) + 1.0

    skellam_mu1 = (var_diff + mu_diff) / 2
    skellam_mu2 = (var_diff - mu_diff) / 2

    # Calculate exact discrete probabilities
    p_draw = skellam.pmf(0, skellam_mu1, skellam_mu2)
    p_loss = skellam.cdf(-1, skellam_mu1, skellam_mu2)
    p_win = skellam.sf(0, skellam_mu1, skellam_mu2)

    # Calculate Expected ELO
    expected_elo = (p_win * elo_win) + (p_loss * elo_loss) + (p_draw * elo_draw)

    # Print results
    print(f"Matchup: {player_a['web_name']} (ID: {player_a_id}) vs {player_b['web_name']} (ID: {player_b_id})")
    print(f"Win: {p_win*100:.1f}% | Exact Draw: {p_draw*100:.1f}% | Loss: {p_loss*100:.1f}%")
    print(f"Expected ELO: {expected_elo:+.2f}")

    return expected_elo, p_win, p_loss, p_draw

# --- CELL 45 ---
def generate_all_duels_matrix(df):
    """
    Generates a matrix of Win, Draw, and Loss probabilities for all possible
    player matchups where Perf_IDX > 0.
    """
    # 1. Filter active players and calculate individual variances
    active_df = df[df['Perf_IDX'] > 0].copy()
    active_df['variance'] = np.maximum((active_df['ceiling_score'] - active_df['Perf_IDX']) / 1.645, 1.0) ** 2

    # 2. Create a cross-join of all possible matchups (Player A vs Player B)
    cols_to_keep = ['id_player', 'web_name', 'Perf_IDX', 'variance']
    duels = pd.merge(
        active_df[cols_to_keep],
        active_df[cols_to_keep],
        how='cross',
        suffixes=('_A', '_B')
    )

    # 3. Remove self-matchups (e.g., Salah vs Salah)
    duels = duels[duels['id_player_A'] != duels['id_player_B']].copy()

    # 4. Calculate Skellam parameters via vectorized NumPy operations
    mu_diff = duels['Perf_IDX_A'] - duels['Perf_IDX_B']
    var_diff = duels['variance_A'] + duels['variance_B']

    # Ensure variance is strictly greater than the absolute mean difference
    var_diff = np.where(var_diff <= np.abs(mu_diff), np.abs(mu_diff) + 0.1, var_diff)

    sk_mu1 = (var_diff + mu_diff) / 2
    sk_mu2 = (var_diff - mu_diff) / 2

    # 5. Calculate probabilities
    duels['Win_%'] = skellam.sf(0, sk_mu1, sk_mu2) * 100
    duels['Draw_%'] = skellam.pmf(0, sk_mu1, sk_mu2) * 100
    duels['Loss_%'] = skellam.cdf(-1, sk_mu1, sk_mu2) * 100

    # Select and format final columns
    final_cols = ['id_player_A','web_name_A','id_player_B', 'web_name_B', 'Win_%', 'Draw_%', 'Loss_%', 'Perf_IDX_A', 'Perf_IDX_B']

    return duels[final_cols].sort_values('Win_%', ascending=False).reset_index(drop=True)

# --- CELL 46 ---
def minutes_composite_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_weight: np.ndarray = None,
    w_huber: float = 0.40,
    w_played: float = 0.35,
    w_sixty: float = 0.25,
    delta: float = 15.0,
) -> float:
    """
    Three-component loss for minutes prediction.

    Component 1 — Huber (continuous accuracy, lower weight than before)
    Component 2 — Played penalty: misclassifying DNP vs played
    Component 3 — 60-min penalty: misclassifying 1pt vs 2pt appearance
    """
    if sample_weight is None:
        sample_weight = np.ones_like(y_true, dtype=float)

    error = np.abs(y_true - y_pred)

    # 1. Huber — continuous regression accuracy
    is_small = error <= delta
    huber_per_obs = np.where(is_small, 0.5 * error**2, delta * (error - 0.5 * delta))
    huber = np.average(huber_per_obs, weights=sample_weight)

    # 2. Played threshold — binary: did model correctly identify DNP (actual=0)?
    #    Penalty = predicted minutes when true = 0 (false positive playing time)
    #    + true minutes when predicted = 0 but player actually played (false negative)
    dnp_mask    = y_true == 0
    played_mask = y_true > 0

    played_penalty = np.average(y_pred[dnp_mask], weights=sample_weight[dnp_mask]) if dnp_mask.any() else 0.0
    missed_penalty = np.average(y_true[played_mask & (y_pred == 0)], weights=sample_weight[played_mask & (y_pred == 0)]) if (played_mask & (y_pred == 0)).any() else 0.0
    played_loss = (played_penalty + missed_penalty) / 90.0   # normalise to [0,1] scale

    # 3. Sixty-minute threshold — binary: did we get the 1pt vs 2pt boundary right?
    #    Only evaluate on players who actually played (exclude DNPs)
    sub_sixty_true = (y_true[played_mask] < 60).astype(float)
    sub_sixty_pred = (y_pred[played_mask] < 60).astype(float)
    sixty_loss = np.average(np.abs(sub_sixty_true - sub_sixty_pred), weights=sample_weight[played_mask]) if played_mask.any() else 0.0

    return w_huber * huber + w_played * played_loss + w_sixty * sixty_loss


# --- CELL 47 ---
def asymmetric_mae(y_true, y_pred, under_penalty=1.5):
    """
    Calculates MAE but penalizes under-predictions more heavily.
    This forces the model to respect high-upside hauls.
    """
    error = np.array(y_true) - np.array(y_pred)
    # error > 0 means actual > predicted (missed haul)
    weighted_error = np.where(error > 0, error * under_penalty, np.abs(error))
    return np.mean(weighted_error)

def calculate_overall_score(df, target_col, pred_col, top_k=50):
    """
    Calculates a composite score optimizing for top-tier ranking (80%)
    and realistic point scaling (20%). Evaluated per gameweek to handle blanks and doubles.
    """
    mask = (df[pred_col] > 0.1)
    # Ensure 'gameweek' is available for groupby operations
    clean_df = df[mask].dropna(subset=[target_col, pred_col, 'gameweek']).copy()

    if len(clean_df) < 50:
        return 0.0, 10.0

    # 1. Global Ranking Quality (Per-GW NDCG)
    ndcg_scores = []

    # 2. Error on Top Tier (Per-GW Rank Percentiles)
    clean_df['pred_rank'] = clean_df.groupby('gameweek')[pred_col].rank(pct=True)
    clean_df['actual_rank'] = clean_df.groupby('gameweek')[target_col].rank(pct=True)

    for gw, gw_df in clean_df.groupby('gameweek'):
        if len(gw_df) >= top_k:
            min_target = min(0, gw_df[target_col].min())
            shifted_targets = gw_df[target_col].values - min_target

            true_scores = np.asarray([shifted_targets])
            pred_scores = np.asarray([gw_df[pred_col].values])

            gw_ndcg = ndcg_score(true_scores, pred_scores, k=top_k)
            ndcg_scores.append(gw_ndcg)

    ranking_score = np.mean(ndcg_scores) if ndcg_scores else 0.0

    # Filter top tier using the per-GW percentiles
    top_tier = clean_df[(clean_df['pred_rank'] >= 0.75) | (clean_df['actual_rank'] >= 0.75)]

    if len(top_tier) > 10:
        error_metric = asymmetric_mae(top_tier[target_col], top_tier[pred_col], under_penalty=1.5)
    else:
        error_metric = 10.0

    point_spread = top_tier[target_col].std()

    if np.isnan(point_spread) or point_spread < 1.0:
        point_spread = 10.0

    normalized_mae = min(error_metric / point_spread, 1.0)

    composite_score = (0.80 * ranking_score) - (0.20 * normalized_mae)

    return composite_score

# --- CELL 48 ---
def get_averaged_production_params(study, top_k=5, primary_metric_idx=0, maximize_primary=True):
    """
    Averages parameters for either Single-Objective or Multi-Objective studies.
    """
    # 1. Detect Study Type
    is_multi_objective = len(study.directions) > 1

    # 2. Extract and Sort Trials Appropriately
    if is_multi_objective:
        # Multi-Objective: Use the Pareto front
        valid_trials = study.best_trials
        if not valid_trials:
            print("No Pareto trials found.")
            return None
        valid_trials.sort(key=lambda t: t.values[primary_metric_idx], reverse=maximize_primary)

        print(f"\n--- ENSEMBLE AVERAGING (Top {min(top_k, len(valid_trials))} Pareto Trials) ---")
        print(f"Best Primary Metric: {valid_trials[0].values[primary_metric_idx]:.4f}")

    else:
        # Single-Objective: Use all complete trials
        valid_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not valid_trials:
            print("No complete trials found.")
            return None
        valid_trials.sort(key=lambda t: t.value, reverse=maximize_primary)

        print(f"\n--- ENSEMBLE AVERAGING (Top {min(top_k, len(valid_trials))} Trials) ---")
        print(f"Best Score: {valid_trials[0].value:.4f}")

    # 3. Select Top K
    k = min(top_k, len(valid_trials))
    top_trials = valid_trials[:k]

    # 4. Accumulate Sums & Calculate Averages
    param_sums, param_counts, avg_params = {}, {}, {}

    for t in top_trials:
        for p_name, p_val in t.params.items():
            param_sums[p_name] = param_sums.get(p_name, 0.0) + p_val
            param_counts[p_name] = param_counts.get(p_name, 0) + 1

    for p_name in param_sums:
        avg_params[p_name] = param_sums[p_name] / param_counts[p_name]

    return avg_params

# --- CELL 51 ---
# --- Main Configuration ---
MY_FPL_ID = 6025459
gameweek_summary = get_fpl_gameweek_summary()

# --- CELL 52 ---
# Get the current gameweek (GW) where 'Is Current' is True
current_gameweek_row = gameweek_summary[gameweek_summary['Is Current'] == True]
if not current_gameweek_row.empty:
    GW = current_gameweek_row['Gameweek ID'].iloc[0]
else:
    GW = None # Or set a default value or raise an error if no current gameweek is found
    print("No current gameweek found in gameweek_summary.")

# Get the maximum finished gameweek (Data Game week)
finished_gameweeks = gameweek_summary[gameweek_summary['Finished'] == True]
if not finished_gameweeks.empty:
    data_gameweek = finished_gameweeks['Gameweek ID'].max()
else:
    data_gameweek = None # Or set a default value
    print("No finished gameweeks found in gameweek_summary.")

# Fetch the user's current team IDs for the determined GW
if GW is not None:
    MY_CURRENT_TEAM_IDS = get_my_player_ids(MY_FPL_ID, GW)
else:
    MY_CURRENT_TEAM_IDS = []
    print("Cannot fetch team IDs as the current gameweek (GW) is not determined.")

print(f"Current Gameweek (GW): {GW}")
print(f"Maximum Finished Gameweek (Data Game week): {data_gameweek}")

# --- CELL 56 ---
bank_values = 1.1
current_free_transfer_avaliable = 1
max_diff_weight = 0.13
max_upside_weight = 0.12
prior_weight = 10.0

# Constants — tune these to your philosophy
TARGET_LOW  = 90  # bottom of your target zone
TARGET_HIGH = 99   # top of your target zone

# --- CELL 57 ---
current_realizable_value_dict = {
      # GKP
      139 :4.5,
      736:5.6,

      # DEF
      5:6.9,
      77:4.0,
      151:4.5,
      256:5.8,
      343:4.3,

      #MID
      82:8.1,
      235:10.4,
      237:6.5,
      449:9.9,
      457:5.7,

      #FWD
      100:4.6,
      136:7.2,
      430:14.3,
}

# --- CELL 58 ---
locked_values = sum(current_realizable_value_dict.values()) + bank_values
try:
    fpl_gameweek_data = get_fpl_gameweek_data(MY_FPL_ID)
    weights = get_dynamic_weights(
        fpl_gameweek_data, data_gameweek,
        max_diff_weight=max_diff_weight,
        max_upside_weight=max_upside_weight,
        target_low = TARGET_LOW,
        target_high = TARGET_HIGH,
    )
except Exception as e:
    print(f"⚠️ Weight fetch failed: {e} — using max weights")
    weights = {
        'diff_weight':   max_diff_weight,
        'upside_weight': max_upside_weight,
        'decay_factor':  1.0,
        'rank_pct':      50.0,
        'mode':          'BALANCED ⚖️ (fallback)',
        'source':        'fallback',
    }

diff_weight   = weights['diff_weight']
upside_weight = weights['upside_weight']

print(f"My Net Transfer Value : {locked_values:.1f} M")
print(f"Rank Percentile       : {weights['rank_pct']:.1f}%")
print(f"Mode                  : {weights['mode']}")
print(f"Decay Factor          : {weights['decay_factor']:.3f}")
print(f"Diff Weight           : {diff_weight:.4f}  (max {max_diff_weight})")
print(f"Upside Weight         : {upside_weight:.4f}  (max {max_upside_weight})")
print(f"Source                : {weights['source']}")

# --- CELL 59 ---
# =========================================================================
# SEASON-ADAPTIVE PARAMETER ENGINE
# Automatically scales trust and decay parameters based on the current Gameweek
# =========================================================================

LOCKED_PARAMS = {
    # Fixture multiplier structure
    'fixture_alpha_att'           : 0.09,
    'fixture_alpha_def'           : 0.07,
    'blend_alpha'                 : 0.50,
    'min_fixtures_full_trust'     : 15,

    # Score calculation structure
    'cs_clip_lower'               : 0.10,
    'cs_clip_upper'               : 0.765,
    'finishing_factor_clip_lower' : 0.50,
    'finishing_factor_clip_upper' : 1.69,
    'protective_factor_clip_lower': 0.65,
    'protective_factor_clip_upper': 1.50,

    # Minutes engine structure
    'minutes_role_floor'          : 0.485,   # UPDATED (from 0.42)
    'minutes_loyalty_w'           : 0.475,   # UPDATED (from 0.95)
    'minutes_trend_scale'         : 0.10,
    'minutes_high_streak'         : 2.4,     # UPDATED (from 3)
    'minutes_low_vol_thresh'      : 5.0,     # UPDATED (from 45.0)

    # League prior
    'league_avg_xG'               : 1.45,
    'league_avg_xGC'              : 1.45,
}

def get_adaptive_params(current_gw: int, locked_params: dict) -> dict:
    """Linearly interpolates confidence parameters between GW1 and GW20."""
    TRANSITION_GW = 20
    # Guard against None GW values (defaults to late-season if missing)
    if current_gw is None:
        current_gw = 38

    t = min(current_gw / TRANSITION_GW, 1.0)   # 0.0 at GW1, 1.0 at GW20+

    def lerp(start, end):
        return round(start + t * (end - start), 4)

    adaptive = {
        'c_finish'         : lerp(30.0,  0.5),     # GW1: Strong prior -> GW20: Trust individual
        'c_protect'        : lerp(30.0,  7.815),   # UPDATED (steady state 7.815)
        'c_base_defense'   : lerp(20.0,  8.0),
        'recency_ema_alpha': lerp(0.30,  0.00),
        'rolling_ema_alpha': lerp(0.10,  0.33),    # UPDATED (steady state 0.33)
        'fixture_alpha_att': lerp(0.04,  0.09),
        'fixture_alpha_def': lerp(0.03,  0.07),
    }
    return {**locked_params, **adaptive}

def get_minutes_params(current_gw: int) -> dict:
    """Phases the Minutes Engine from baseline-heavy (Early) to form-heavy (Late)."""
    if current_gw is None:
        current_gw = 38

    if current_gw <= 6:
        # EARLY: No meaningful form signal exists yet.
        return {
            'minutes_w_form'          : 0.40,
            'minutes_w_haaland_season': 0.60,
            'minutes_w_gk_form'       : 0.70,
            'minutes_ema_alpha'       : 0.30,
        }
    elif current_gw <= 20:
        # MID: Linear interpolation across the zone.
        t = (current_gw - 6) / (20 - 6)
        def lerp(start, end):
            return round(start + t * (end - start), 3)

        return {
            'minutes_w_form'          : lerp(0.40, 0.935),  # UPDATED (target 0.935)
            'minutes_w_haaland_season': lerp(0.60, 0.155),  # UPDATED (target 0.155)
            'minutes_w_gk_form'       : lerp(0.70, 1.00),
            'minutes_ema_alpha'       : lerp(0.30, 0.925),  # UPDATED (target 0.925)
        }
    else:
        # LATE: Squads settled, form dominates.
        return {
            'minutes_w_form'          : 0.935,  # UPDATED
            'minutes_w_haaland_season': 0.155,  # UPDATED
            'minutes_w_gk_form'       : 1.00,
            'minutes_ema_alpha'       : 0.925,  # UPDATED
        }

def get_season_params(current_gw: int) -> dict:
    base    = get_adaptive_params(current_gw, LOCKED_PARAMS)
    minutes = get_minutes_params(current_gw)
    return {**base, **minutes}

# Generate the params dynamically for the current Gameweek
params = get_season_params(GW)

# --- CELL 60 ---
start_time = time.time()

player_df         = get_current_players_df()
active_player_ids = player_df['id'].unique()

raw_history_df = await fetch_raw_history_cache(active_player_ids, use_cache=True)

fpl_team_df = get_team_df()

rolling_ratings_raw, latest_ratings_raw = compute_rolling_team_ratings(
    raw_history_df,
    ema_alpha    = params.get('rolling_ema_alpha', 0.15),
    min_fixtures = 3,
)

team_ratings_df, latest_team_ratings = blend_team_ratings(
    rolling_ratings_raw,
    latest_ratings_raw,
    fpl_team_df,
    league_avg_xG           = params.get('league_avg_xG',            1.45),
    league_avg_xGC          = params.get('league_avg_xGC',           1.45),
    blend_alpha             = params.get('blend_alpha',              0.75),
    min_fixtures_full_trust = params.get('min_fixtures_full_trust',  10),
)

global_dists = compute_global_z_distributions(team_ratings_df)

fixture_player_df = get_fixture_players_stats_df(
    params,
    raw_history_df,
    global_dists,
    team_ratings_df    = team_ratings_df,
    latest_team_ratings= latest_team_ratings,
)

reg_params  = _fit_regression_params(fixture_player_df)
bonus_model = _fit_bonus_multinomial(raw_history_df)
_diagnose_bonus_model(bonus_model)
params.update(reg_params)

fixture_player_df = _calculate_performance_indices(
    fixture_player_df,
    params,
    bonus_model=bonus_model
)

gc.collect()
print(f"CPU times: Wall time: {time.time() - start_time:.2f} s")

# --- CELL 62 ---
# Grouping structure
grouping_columns = [
    'gameweek', 'id_player', 'now_cost', 'selected_by_percent',
    'web_name', 'position', 'team_name'
]

# Sum columns (accumulated values over fixtures)
sum_columns = [
    'Perf_IDX', 'ceiling_score', 'GOAL_INDEX', 'ASSIST_INDEX',
    'CLEAN_SHEET_INDEX', 'bonus_component', 'defcon_component',
    'minutes_IDX', 'actual_minutes'
]

# Mean columns (rate metrics and constants)
mean_columns = [
    'recent_minutes_form', 'finishing_factor', 'protective_factor',
    'fixture_attack_multiplier', 'fixture_defence_multiplier',
    'fixture_calibrated_points',
    'start_per_gameplayed', 'consecutive_start_streak','hybrid_bps_abs','score_std',
]

agg_dict = {col: 'sum' for col in sum_columns}
agg_dict.update({col: 'mean' for col in mean_columns})

display_columns = [
    'id_player', 'now_cost', 'selected_by_percent', 'web_name', 'position',
    'team_name', 'minutes_IDX', 'Perf_IDX', 'ceiling_score',
    'GOAL_INDEX', 'ASSIST_INDEX', 'CLEAN_SHEET_INDEX', 'bonus_component',
    'defcon_component', 'finishing_factor', 'fixture_attack_multiplier',
    'protective_factor', 'fixture_defence_multiplier',
]

# Execute Pipeline
gw_projection_df = fixture_player_df.groupby(grouping_columns).agg(agg_dict).reset_index()

gw_projection_df = create_optimized_custom_score(
    df=gw_projection_df,
    differential_weight=diff_weight,
    upside_weight=upside_weight,
    # visualize=True
)

# Render Output
display_columns_final = display_columns + ['custom_score', 'ceiling_score', 'dynamic_upside']

# --- CELL 64 ---
projection_group = ['id_player', 'now_cost', 'selected_by_percent', 'web_name',
       'position', 'team_name', ]
avg_columns = list(set(list(agg_dict.keys()) + ['custom_score', 'ceiling_score', 'dynamic_upside']))

# --- CELL 65 ---
