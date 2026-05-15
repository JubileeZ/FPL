# fpl_engine/config.py
# =========================================================================
# Static configuration and tuned hyperparameters.
# This module must NOT make API calls at import time — keep it pure constants.
# Runtime setup (gameweek detection, team fetching, dynamic weights) belongs
# in the notebook or a dedicated setup function.
# =========================================================================

# --- CELL 56 ---
MY_FPL_ID = 6025459

# --- CELL 59 ---
# =========================================================================
# SEASON-ADAPTIVE PARAMETER ENGINE
# Automatically scales trust and decay parameters based on the current Gameweek
# =========================================================================
import json
import os
from datetime import datetime

# =========================================================================
# SEASON-ADAPTIVE PARAMETER ENGINE
# Automatically scales trust and decay parameters based on the current Gameweek
# =========================================================================

def load_tuned_params():
    """Loads tuned parameters from the JSON file."""
    # Find the path to the root directory's tuned_params.json
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'tuned_params.json')
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback to safe defaults if the file doesn't exist
        print("WARNING: tuned_params.json not found! Using fallback defaults.")
        return {
            "locked_params": {
                'fixture_alpha_att': 0.09, 'fixture_alpha_def': 0.07, 'blend_alpha': 0.50,
                'min_fixtures_full_trust': 15, 'cs_clip_lower': 0.10, 'cs_clip_upper': 0.765,
                'finishing_factor_clip_lower': 0.50, 'finishing_factor_clip_upper': 1.69,
                'protective_factor_clip_lower': 0.65, 'protective_factor_clip_upper': 1.50,
                'minutes_role_floor': 0.485, 'minutes_loyalty_w': 0.475, 'minutes_trend_scale': 0.10,
                'minutes_high_streak': 2.4, 'minutes_low_vol_thresh': 5.0,
                'league_avg_xG': 1.45, 'league_avg_xGC': 1.45
            },
            "adaptive_targets": {
                'c_finish': 0.5, 'c_protect': 7.815, 'c_base_defense': 8.0,
                'recency_ema_alpha': 0.00, 'rolling_ema_alpha': 0.33,
                'fixture_alpha_att': 0.09, 'fixture_alpha_def': 0.07
            },
            "minutes_targets": {
                'minutes_w_form': 0.935, 'minutes_w_haaland_season': 0.155,
                'minutes_w_gk_form': 1.00, 'minutes_ema_alpha': 0.925
            }
        }

def get_adaptive_params(current_gw: int, locked_params: dict, targets: dict) -> dict:
    """Linearly interpolates confidence parameters between GW1 and GW20."""
    TRANSITION_GW = 20
    # Guard against None GW values (defaults to late-season if missing)
    if current_gw is None:
        current_gw = 38

    t = min(current_gw / TRANSITION_GW, 1.0)   # 0.0 at GW1, 1.0 at GW20+

    def lerp(start, end):
        return round(start + t * (end - start), 4)

    adaptive = {
        'c_finish'         : lerp(30.0, targets['c_finish']),     
        'c_protect'        : lerp(30.0, targets['c_protect']),   
        'c_base_defense'   : lerp(20.0, targets['c_base_defense']),
        'recency_ema_alpha': lerp(0.30, targets['recency_ema_alpha']),
        'rolling_ema_alpha': lerp(0.10, targets['rolling_ema_alpha']),   
        'fixture_alpha_att': lerp(0.04, targets['fixture_alpha_att']),
        'fixture_alpha_def': lerp(0.03, targets['fixture_alpha_def']),
    }
    return {**locked_params, **adaptive}


def get_minutes_params(current_gw: int, targets: dict) -> dict:
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
            'minutes_w_form'          : lerp(0.40, targets['minutes_w_form']), 
            'minutes_w_haaland_season': lerp(0.60, targets['minutes_w_haaland_season']), 
            'minutes_w_gk_form'       : lerp(0.70, targets['minutes_w_gk_form']),
            'minutes_ema_alpha'       : lerp(0.30, targets['minutes_ema_alpha']),
        }
    else:
        # LATE: Squads settled, form dominates.
        return targets.copy()


def get_season_params(current_gw: int) -> dict:
    """Combines LOCKED_PARAMS + adaptive interpolation + minutes phasing for a given GW."""
    config_data = load_tuned_params()
    base    = get_adaptive_params(current_gw, config_data['locked_params'], config_data['adaptive_targets'])
    minutes = get_minutes_params(current_gw, config_data['minutes_targets'])
    return {**base, **minutes}
def get_advanced_model_config() -> dict:
    """Returns the advanced_model section with safe defaults if missing."""
    config_data = load_tuned_params()
    return config_data.get("advanced_model", {
        "enable_covariance_ceiling": False,
        "enable_garch_minutes": False,
        "enable_scenarios": False,
        "scenario_count": 5000,
        "cvar_alpha": 0.10,
        "cvar_weight": 0.0,
        "column_gen_top_k": 80
    })
