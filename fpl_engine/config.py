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
