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

