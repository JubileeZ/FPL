# fpl_engine/config.py
# =========================================================================
# Static configuration and tuned hyperparameters.
# This module must NOT make API calls at import time — keep it pure constants.
# Runtime setup (gameweek detection, team fetching, dynamic weights) belongs
# in the notebook or a dedicated setup function.
# =========================================================================

# --- CELL 56 ---
MY_FPL_ID = 6025459

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
    """Combines LOCKED_PARAMS + adaptive interpolation + minutes phasing for a given GW."""
    base    = get_adaptive_params(current_gw, LOCKED_PARAMS)
    minutes = get_minutes_params(current_gw)
    return {**base, **minutes}

