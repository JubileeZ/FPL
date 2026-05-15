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

def _fit_regression_params(
    fixture_player_df: pd.DataFrame,
) -> dict:
    df = fixture_player_df.copy()

    # --- Goals ---
    comb_g = df[['expected_goals', 'threat', 'goals_scored']].dropna()
    model_goals = LinearRegression(fit_intercept=False, positive=True).fit(
        comb_g[['expected_goals', 'threat']], comb_g['goals_scored']
    )

    # --- Assists ---
    comb_a = df[['expected_assists', 'creativity', 'assists']].dropna()
    model_assists = LinearRegression(fit_intercept=False, positive=True).fit(
        comb_a[['expected_assists', 'creativity']], comb_a['assists']
    )

    return {
        'w_xG':        model_goals.coef_[0],
        'w_threat':    model_goals.coef_[1],
        'w_xA':        model_assists.coef_[0],
        'w_creativity': model_assists.coef_[1],
    }

# --- CELL 30 ---
def _fit_bonus_multinomial(raw_history_df: pd.DataFrame, min_minutes: int = 45) -> LogisticRegression:
    """ Fit P(bonus ∈ {0,1,2,3} | absolute bps) on per-fixture observations.
    Returns a fitted sklearn LogisticRegression with .classes = [0, 1, 2, 3].
    """
    hist = raw_history_df[raw_history_df['minutes'] >= min_minutes].copy()  # ← starters only
    hist = hist[['bps', 'bonus']].dropna()
    hist = hist[hist['bps'] >= 0]
    hist['bonus'] = hist['bonus'].clip(0, 3).astype(int)

    # Fit Model
    bonus_model = LogisticRegression()
    bonus_model.fit(hist[['bps']], hist['bonus'])

    # [PATCH]: Moved BPS monotonicity validation check BEFORE the return statement
    test_bps = np.array([[10], [20], [30], [40], [50]])
    probs = bonus_model.predict_proba(test_bps)
    expected_values = [np.dot(p, bonus_model.classes_) for p in probs]

    is_monotonic = all(expected_values[i] <= expected_values[i+1] for i in range(len(expected_values)-1))
    if not is_monotonic:
        import warnings
        warnings.warn("CRITICAL: BPS to Bonus multinomial curve is not strictly monotonic! "
                      "Consider post-fit isotonic regression calibration.")

    return bonus_model

# --- CELL 31 ---
def fit_bps_calibration(
    df: pd.DataFrame,
    estimate_col: str = 'estimate_bps_calibrate',
    actual_col: str = 'actual_avg_bps',
    min_minutes: int = 45,
) -> dict:
    """
    Fits per-position linear calibration: actual_bps ~ estimate_bps.
    Call once on bootstrap output (train data only). Returns frozen calibrators.
    """
    sub = df[df['actual_minutes'] >= min_minutes].copy()
    sub = sub[['position', estimate_col, actual_col]].dropna()
    sub = sub[sub[actual_col] >= 0]

    calibrators = {}
    for pos in ['GKP', 'DEF', 'MID', 'FWD']:
        pos_df = sub[sub['position'] == pos]
        if len(pos_df) < 50:
            print(f"WARNING: {pos} has only {len(pos_df)} rows — skipping, will use fallback.")
            continue
        X = pos_df[[estimate_col]].values
        y = pos_df[actual_col].values
        model = LinearRegression().fit(X, y)
        calibrators[pos] = model
        print(f"{pos}: scale={model.coef_[0]:.3f}  intercept={model.intercept_:.3f}  "
              f"R²={model.score(X, y):.3f}  n={len(pos_df)}")

    return calibrators


def _apply_bps_calibration(
    estimate_series: pd.Series,
    positions: pd.Series,
    calibrators: dict,
) -> pd.Series:
    """
    Applies per-position linear calibration to a BPS estimate Series.
    Falls back to average scale/intercept for unknown positions.
    """
    result = pd.Series(np.nan, index=estimate_series.index)

    for pos, model in calibrators.items():
        mask = positions == pos
        if mask.any():
            result[mask] = np.clip(
                model.predict(estimate_series[mask].values.reshape(-1, 1)).flatten(),
                0, None
            )

    missing = result.isna()
    if missing.any():
        avg_coef      = np.mean([m.coef_[0] for m in calibrators.values()])
        avg_intercept = np.mean([m.intercept_ for m in calibrators.values()])
        result[missing] = np.clip(
            avg_coef * estimate_series[missing] + avg_intercept,
            0, None
        )

    return result

# --- CELL 32 ---
# --- Diagnostics (run once after fitting) ---
def _diagnose_bonus_model(bonus_model: LogisticRegression):
    print(f"{'bps':>6}  {'P(0)':>6}  {'P(1)':>6}  {'P(2)':>6}  {'P(3)':>6}  {'E(bonus)':>8}")
    for bps in [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]:
        p = bonus_model.predict_proba([[bps]])[0]
        ev = float(p @ bonus_model.classes_.astype(float))
        print(f"{bps:>6}  {p[0]:>6.3f}  {p[1]:>6.3f}  {p[2]:>6.3f}  {p[3]:>6.3f}  {ev:>8.3f}")

# --- CELL 34 ---
def _calculate_performance_indices(
    fixture_player_df: pd.DataFrame,
    params: dict,
    bonus_model: LogisticRegression,
    bps_calibrators: dict = None,
    minute_overrides: dict = None,
    UPSIDE_Z: float = 1.5,
):
    """
    Calculates FPL performance indices for every player-fixture row.
    Regression weights must be pre-injected into params.

    Produces three core output columns:
      - Perf_IDX               : expected points (probability-weighted mean outcome)
      - fixture_calibrated_pts : Perf_IDX re-run on actual minutes (for model validation)
      - ceiling_score          : upside scenario using CORRECT variance aggregation.
                                 ceiling = Perf_IDX + UPSIDE_Z * √(Σ component variances)
                                 This replaces the old approach that added √var per component
                                 separately, which incorrectly assumed all ceiling events
                                 (peak goals, peak CS, peak saves) co-occur in the same match.

    Parameters
    ----------
    fixture_player_df : player-fixture projection dataframe
    params            : dict of model hyperparameters (weights, clips, thresholds)
    bonus_model       : fitted LogisticRegression — maps BPS → P(0,1,2,3 bonus pts)
    bps_calibrators   : optional dict of per-position BPS scalers.
                        If None, the function fits them internally from actual_minutes rows.
    minute_overrides  : optional dict of {player_id: minutes} or {gw: {player_id: minutes}}
                        to manually override the minutes engine for specific players/GWs.
    UPSIDE_Z          : float — ceiling = mean + Z * std.
                        1.0 = 84th percentile, 1.5 = 93rd percentile (recommended), 2.0 = 98th.
    """

    if minute_overrides is None:
        minute_overrides = {}

    p  = params.copy()
    df = fixture_player_df.copy()

    # =========================================================================
    # PART A — PRE-CALCULATIONS (PER-90 RATES)
    # Compute per-90 rates and player-level multipliers.
    # These are fixture-agnostic calibrations that anchor expected vs actual
    # performance — applied to minutes projections later.
    # =========================================================================

    C_FINISH  = p.get('c_finish', 5.0)
    C_PROTECT = p.get('c_protect', 15.0)
    C         = p.get('c_base_defense', 5.0)

    # --- A1. Finishing Factor ---
    # Measures whether this player outperforms (>1) or underperforms (<1) their xG.
    # Bayesian shrinkage constant C_FINISH pulls extreme values toward 1.0
    # to avoid overweighting small sample flukes.
    # Clipped to prevent outliers from dominating the goal projection.
    df['indiv_finishing_factor'] = (
        (df['goals_scored'] + C_FINISH) / (df['expected_goals'] + C_FINISH)
    ).clip(
        lower=p['finishing_factor_clip_lower'],
        upper=p['finishing_factor_clip_upper']
    )
    finishing_mult = df['indiv_finishing_factor']

    # --- A2. Protective Factor (GKP/DEF only) ---
    # Measures whether this player's team concedes fewer goals than xGC predicts.
    # Outfield players are set to 1.0 (no adjustment) — they do not affect
    # the defensive structure independently.
    df['raw_protective_factor'] = (
        (df['goals_conceded'] + C_PROTECT) / (df['expected_goals_conceded'] + C_PROTECT)
    ).clip(
        lower=p['protective_factor_clip_lower'],
        upper=p['protective_factor_clip_upper']
    )
    df['indiv_protective_factor'] = np.where(
        df['position'].isin(['GKP', 'DEF']),
        df['raw_protective_factor'],
        1.0
    )
    protective_mult = df['indiv_protective_factor']

    # --- A3. Expected Goals Conceded Per 90 (adj_xGC_90) ---
    # Blends player-level xGC with team-level xGC per 90 using a Bayesian
    # prior weight C (default 5). This stabilises the rate for players with
    # fewer historical minutes while still reflecting their individual record.
    # Then adjusted by fixture difficulty and the player's protective factor.
    xGC_90_base = (
        (df['expected_goals_conceded'] + (C * df['team_expected_goals_conceded_per_90']))
        / (df['minutes'] + (C * 90))
    ) * 90
    adj_xGC_90 = np.maximum(
        xGC_90_base * df['fixture_defence_multiplier'] * protective_mult,
        0.001   # floor to avoid log(0) in CS probability calculations
    )

    # --- A4. Attacking Rates (Hybrid xG/xA per 90) ---
    # Blends two signals for each attacking metric:
    #   - Creativity/Threat: FPL's proprietary in-match contribution metric
    #   - xA/xG:             StatsBomb expected values from chance quality
    # Fixture attack multiplier adjusts for opponent defensive strength.
    hybrid_xA_90 = (
        p['w_creativity'] * df['creativity_per_90'].fillna(0)
        + p['w_xA']       * df['expected_assists_per_90'].fillna(0)
    ) * df['fixture_attack_multiplier']

    hybrid_xG_90 = (
        p['w_threat'] * df['threat_per_90'].fillna(0)
        + p['w_xG']   * df['expected_goals_per_90'].fillna(0)
    ) * df['fixture_attack_multiplier']

    # --- A5. Points and BPS Scalars by Position ---
    # FPL official points values per action, mapped by position.
    # Also maps the equivalent BPS weights (for bonus point model input).
    minutes_mask_bonus  = 3                                                          # BPS per appearance / per 60-min appearance
    pos_mask_cs         = df['position'].map({'GKP': 4,  'DEF': 4,  'MID': 1, 'FWD': 0}).fillna(0)   # pts per CS
    pos_mask_cs_bonus   = df['position'].map({'GKP': 12, 'DEF': 12, 'MID': 0, 'FWD': 0}).fillna(0)   # BPS per CS
    pos_mask_goal       = df['position'].map({'GKP': 10, 'DEF': 6,  'MID': 5, 'FWD': 4}).fillna(0)   # pts per goal
    pos_mask_goal_bonus = df['position'].map({'GKP': 12, 'DEF': 12, 'MID': 18, 'FWD': 24}).fillna(0) # BPS per goal
    penalty_mask        = df['position'].map({'GKP': -1, 'DEF': -1}).fillna(0)      # pts per goal conceded
    penalty_mask_bonus  = df['position'].map({'GKP': -4, 'DEF': -4}).fillna(0)      # BPS per goal conceded
    assist_mask         = 3     # pts per assist (all positions)
    assist_mask_bonus   = 9     # BPS per assist
    yellow_card_mask    = -1    # pts per yellow card
    yellow_card_mask_bonus = -3 # BPS per yellow card
    red_card_mask       = -3    # pts per red card
    red_card_mask_bonus = -9    # BPS per red card
    defcon_mask         = 2     # pts per defensive bonus event (clearance/block/interception)
    defcon_mask_bonus   = 1/2   # BPS per defensive action (threshold-triggered)
    save_mask           = 1/3   # pts per save (GKP: 1pt per 3 saves)
    save_mask_bonus     = 2.75  # BPS per save

    # =========================================================================
    # PART B — MINUTES ENGINE
    # Produces df['minutes_IDX']: fixture-adjusted projected minutes per player.
    # This is the key quantity that scales every downstream expected value.
    # =========================================================================

    w_form           = p.get('minutes_w_form', 0.65)
    w_season         = 1.0 - w_form
    w_haaland_season = p.get('minutes_w_haaland_season', 0.90)
    w_haaland_form   = 1.0 - w_haaland_season
    w_gk_form        = p.get('minutes_w_gk_form', 0.80)
    w_gk_season      = 1.0 - w_gk_form
    role_floor       = p.get('minutes_role_floor', 0.85)

    # --- B1. Base Blend ---
    # Default: 65% recent form (last N games) + 35% season average.
    # Weighted toward form to capture current fitness and rotation trends.
    base_minutes = (
        w_form   * df['recent_minutes_form'].fillna(0)
        + w_season * df['minutes_per_game'].fillna(0)
    )

    # Boolean flags used in multiple conditions below
    is_fully_fit      = df['chance_of_playing_next_round'].fillna(100) == 100
    recent_dip        = df['recent_minutes_form'].fillna(0) < (df['minutes_per_game'].fillna(0) - 15)
    is_proven_starter = (
        (df['start_per_gameplayed'].fillna(0) > 0.80)
        & (df['start_share_total_game'].fillna(0) > 0.50)
    )
    is_outfielder = df['position'] != 'GKP'

    # --- B2. Proven Starter / GKP Dropped Correction ---
    # A star outfielder (>80% start rate, fully fit) with a recent dip in minutes
    # is more likely being managed/rested than genuinely dropped.
    # → Reweight toward season average (90%) rather than punishing them for rotation.
    #
    # Same logic for goalkeepers — GKPs are almost never genuinely dropped;
    # a recent dip usually means an injury or cup game, not a permanent change.
    haaland_condition   = is_proven_starter & is_fully_fit & recent_dip & is_outfielder
    gk_dropped_condition = (df['position'] == 'GKP') & is_fully_fit & recent_dip

    base_minutes = np.where(
        haaland_condition,
        w_haaland_form   * df['recent_minutes_form'].fillna(0)
        + w_haaland_season * df['minutes_per_game'].fillna(0),
        np.where(
            gk_dropped_condition,
            w_gk_form   * df['recent_minutes_form'].fillna(0)
            + w_gk_season * df['minutes_per_game'].fillna(0),
            base_minutes
        )
    )

    # --- B2.5 Rest Bounce-Back ---
    # A proven starter who is fully fit and played 0 last game with a high season
    # average (>75 mpg) was almost certainly rested, not dropped.
    # Override strongly toward season average.
    BOUNCE_SEASON_W = p.get('minutes_bounce_season_w', 0.95)
    rest_bounce = (
        is_proven_starter       # >80% start rate & >50% start share
        & is_fully_fit          # COPR = 100
        & (df['last_match_minutes'].fillna(0) == 0)
        & (df['minutes_per_game'].fillna(0) > 75)
    )
    base_minutes = np.where(
        rest_bounce,
        BOUNCE_SEASON_W * df['minutes_per_game'].fillna(0)
        + (1 - BOUNCE_SEASON_W) * 90.0,
        base_minutes
    )

    # Also cap volatility for these rested players so B7 doesn't penalize them
    df['minutes_volatility'] = np.where(
        rest_bounce,
        df['minutes_volatility'].clip(upper=15),
        df['minutes_volatility']
    )

    # --- B3. Manager Loyalty Adjustment ---
    # If a player has low minutes volatility (consistent starter), is on a streak,
    # is fully fit, and played ≥45 last game, anchor strongly to their last match
    # minutes rather than the season average. Captures in-form starters who are
    # being used regularly by their manager.
    LOW_VOL_THRESH = p.get('minutes_low_vol_thresh', 18.0)
    HIGH_STREAK    = p.get('minutes_high_streak', 3)
    LOYALTY_W      = p.get('minutes_loyalty_w', 0.85)

    is_consistent  = df['minutes_volatility'].fillna(30) < LOW_VOL_THRESH
    is_on_streak   = df['consecutive_start_streak'].fillna(0) >= HIGH_STREAK

    loyalty_signal = (
        is_consistent & is_on_streak & is_fully_fit
        & (df['last_match_minutes'].fillna(0) >= 45)
    )

    base_minutes = np.where(
        loyalty_signal,
        LOYALTY_W * df['last_match_minutes'].fillna(0)
        + (1 - LOYALTY_W) * base_minutes,
        base_minutes
    )

    # --- B4. Minutes Trend Adjustment ---
    # If a player's minutes have been trending up or down across recent games
    # (captured by minutes_trend_slope), adjust the projection accordingly.
    # Clipped to ±10 minutes to prevent a single-game outlier from dominating.
    TREND_SCALE = p.get('minutes_trend_scale', 0.5)
    trend_adjustment = np.clip(
        df['minutes_trend_slope'].fillna(0) * TREND_SCALE,
        -10, 10
    )
    base_minutes = base_minutes + trend_adjustment

    # --- B5. Reclaimed Starter Condition ---
    # If base_minutes is projecting low (<65) but the player is fully fit and
    # played ≥60 last game, they have likely reclaimed a starting spot.
    # → Blend heavily toward last match and full game to avoid under-projecting.
    reclaimed_starter_condition = (
        is_fully_fit
        & (df['last_match_minutes'].fillna(0) >= 60)
        & (base_minutes < 65)
    )
    base_minutes = np.where(
        reclaimed_starter_condition,
        0.70 * df['last_match_minutes'].fillna(0) + 0.30 * 90.0,
        base_minutes
    )

    # --- B6. Final Clip ---
    # Hard cap at [0, 90]. No player can be projected for negative or >90 minutes.
    base_minutes = np.clip(base_minutes, 0, 90)

    # --- B7. Volatility-Dampened Role Multiplier ---
    # A player with high minutes volatility (rotation risk) gets a lower
    # effective minutes projection even if their raw average looks good.
    # dynamic_role_floor scales between role_floor (high volatility) and a
    # higher floor (low volatility / consistent starter).
    # Final multiplier also scales with starter_confidence (% of games started).
    starter_confidence = df['start_per_gameplayed'].fillna(0)
    vol_norm           = np.clip(df['minutes_volatility'].fillna(30) / 35.0, 0, 1)
    dynamic_role_floor = role_floor + (1.0 - role_floor) * (1.0 - vol_norm) * 0.5
    role_multiplier    = dynamic_role_floor + ((1.0 - dynamic_role_floor) * starter_confidence)

    # Injury discount: linearly scale down for players with <75% chance of playing,
    # with an additional non-linear penalty for more serious injury doubts.
    injury_prob          = df['chance_of_playing_next_round'].fillna(100) / 100.0
    adjusted_injury_prob = np.where(injury_prob >= 0.75, injury_prob, injury_prob ** 1.5)

    # minutes_IDX: the final fixture-adjusted projected minutes.
    # Used as the primary scaling denominator for all subsequent expected values.
    df['minutes_IDX'] = base_minutes * role_multiplier * adjusted_injury_prob

    # --- B8. Manual Overrides ---
    # Allow callers to override minutes_IDX for specific players or GWs.
    # 'default' key applies to all gameweeks; numeric keys apply per-GW.
    if minute_overrides:
        if 'default' in minute_overrides:
            df['minutes_IDX'] = (
                df['id_player'].map(minute_overrides['default'])
                .fillna(df['minutes_IDX'])
            )
        for gw_key, gw_dict in minute_overrides.items():
            if gw_key == 'default':
                continue
            mask = (df['gameweek'] == gw_key) & (df['id_player'].isin(gw_dict.keys()))
            if mask.any():
                df.loc[mask, 'minutes_IDX'] = df.loc[mask, 'id_player'].map(gw_dict)

    # =========================================================================
    # PART C — EXPECTED VALUE COMPONENTS (Perf_IDX)
    # Scale per-90 rates by minutes_IDX to produce expected points per component.
    # All values are probability-weighted means — the average outcome over many
    # matches with this fixture/minutes projection.
    # =========================================================================

    # --- C1. Expected Goals Conceded (absolute, this fixture) ---
    # Converts the per-90 rate to a per-fixture expectation.
    # Used for CS probability, penalty deductions, and variance calculations.
    adj_xGC_pred = adj_xGC_90 * (df['minutes_IDX'] / 90)

    # --- C2. Clean Sheet Points ---
    # CS probability = P(Poisson(λ) = 0) = exp(-λ) where λ = adj_xGC_pred.
    # Only awarded to players who played ≥60 minutes (FPL rule).
    cs_prob_pred = np.clip(
        np.exp(-adj_xGC_pred),
        a_min=p.get('cs_clip_lower', 0.0),
        a_max=p.get('cs_clip_upper', 1.0)
    )
    df['CLEAN_SHEET_INDEX'] = (
        pos_mask_cs
        * cs_prob_pred
        * (df['minutes_IDX'] >= 60).astype(float)
    )

    # --- C3. Goals Conceded Deduction ---
    # Exact expected deduction formula for GKP/DEF:
    # Accounts for the discrete nature of FPL's -1pt per 2 goals conceded rule.
    # deduction = E[floor(xGC / 2)] computed analytically.
    deduction_project = adj_xGC_pred / 2 - 0.25 + 0.25 * np.exp(-2 * adj_xGC_pred)
    df['CONCEDED_PENALTY'] = penalty_mask * deduction_project

    # --- C4. Assists and Goals ---
    # Scale per-90 rates by projected minutes. Apply position multiplier for goals.
    # finishing_mult adjusts for players who historically over/under-convert chances.
    df['ASSIST_INDEX'] = (hybrid_xA_90 * assist_mask) * (df['minutes_IDX'] / 90)
    df['GOAL_INDEX']   = (hybrid_xG_90 * pos_mask_goal * finishing_mult) * (df['minutes_IDX'] / 90)

    # --- C5. Minutes Points ---
    # FPL awards 1pt for playing (>0 min) and an additional 1pt for playing ≥60 min.
    df['minutes_component'] = (
        (df['minutes_IDX'] > 0).astype(int)
        + (df['minutes_IDX'] >= 60).astype(int)
    )

    # --- C6. Save Points (GKP only) ---
    # save_mask = 1/3 (1 point per 3 saves). Outfield players have save_mask = 0.
    df['SAVE_component'] = (
        df['saves_per_90'] * df['fixture_defence_multiplier'] * save_mask
    ) * (df['minutes_IDX'] / 90)

    # --- C7. Card Penalties ---
    # Expected point deduction from cards, scaled by per-90 rates.
    df['RED_YELLOW_PENALTY'] = (
        (red_card_mask * df['red_cards_index_per_90'])
        + (yellow_card_mask * df['yellow_cards_per_90'])
    ) * (df['minutes_IDX'] / 90)

    # --- C8. Defensive Contribution Bonus (DefCon) ---
    # FPL awards bonus BPS for accumulating clearances, blocks, interceptions.
    # Modelled as P(defensive actions > threshold) using a normal approximation.
    # Mean and std are derived from the player's historical defensive contribution
    # per 90, scaled by minutes projection.
    pred_defcon_abs = (
        df['defensive_contribution_per_90'].fillna(0)
        * df['fixture_defence_multiplier']
        * (df['minutes_IDX'] / 90)
    )
    std_defcon   = np.sqrt(pred_defcon_abs.clip(lower=1e-6))
    defcon_thresh = df['position'].map({'GKP': 100, 'DEF': 10, 'MID': 12, 'FWD': 12}).fillna(100)

    df['defcon_prob']      = stats.norm.sf(defcon_thresh - 0.5, loc=pred_defcon_abs, scale=std_defcon)
    df['defcon_component'] = defcon_mask * df['defcon_prob']

    # --- C9. BPS Estimate (for bonus point model) ---
    # The bonus point classifier maps total expected BPS → P(bonus = 0,1,2,3).
    # This mirrors the BPS formula used by FPL, using expected-value inputs
    # rather than actual counts.
    # Note: penalty term uses xGC directly (intentional approximation for BPS mapping).
    estimate_bps = (
        (df['minutes_IDX'] > 0).astype(int)  * minutes_mask_bonus
        + (df['minutes_IDX'] >= 60).astype(int) * minutes_mask_bonus
        + (hybrid_xG_90 * pos_mask_goal_bonus * finishing_mult) * (df['minutes_IDX'] / 90)
        + (hybrid_xA_90 * assist_mask_bonus) * (df['minutes_IDX'] / 90)
        + (pos_mask_cs_bonus * cs_prob_pred * (df['minutes_IDX'] >= 60).astype(float))
        + (df['saves_per_90'] * df['fixture_defence_multiplier'] * save_mask_bonus) * (df['minutes_IDX'] / 90)
        + (penalty_mask_bonus * adj_xGC_pred)
        + (defcon_mask_bonus * df['defcon_prob'])
        + (red_card_mask_bonus  * df['red_cards_index_per_90']) * (df['minutes_IDX'] / 90)
        + (yellow_card_mask_bonus * df['yellow_cards_per_90'])  * (df['minutes_IDX'] / 90)
    )

    # =========================================================================
    # PART D — BPS CALIBRATION
    # Fits or applies per-position BPS scalers to remove systematic model bias.
    # If no external calibrators are provided, fits them internally from rows
    # where actual minutes are known (≥45 min threshold for reliability).
    # =========================================================================

    actual_mins = df['actual_minutes'].fillna(0)

    # Re-compute all expected components using actual_minutes (not projected minutes_IDX).
    # This allows an apples-to-apples comparison between model predictions and actuals.
    minutes_pts      = (actual_mins > 0).astype(int) + (actual_mins >= 60).astype(int)
    adj_xGC_actual   = adj_xGC_90 * (actual_mins / 90)
    cs_prob_actual   = np.clip(np.exp(-adj_xGC_actual), a_min=p['cs_clip_lower'], a_max=p['cs_clip_upper'])
    cs_pts           = pos_mask_cs * cs_prob_actual * (actual_mins >= 60).astype(float)
    deduction_actual = adj_xGC_actual / 2 - 0.25 + 0.25 * np.exp(-2 * adj_xGC_actual)
    deduction_pts    = penalty_mask * deduction_actual

    pred_defcon_abs_calibrate = (
        df['defensive_contribution_per_90'].fillna(0)
        * df['fixture_defence_multiplier']
        * (actual_mins / 90)
    )
    std_defcon_calibrate  = np.sqrt(pred_defcon_abs_calibrate.clip(lower=1e-6))
    defcon_prob_calibrate = stats.norm.sf(
        defcon_thresh - 0.5,
        loc=pred_defcon_abs_calibrate,
        scale=std_defcon_calibrate
    )
    defcon_pts   = defcon_mask * defcon_prob_calibrate
    assist_pts   = (hybrid_xA_90 * assist_mask) * (actual_mins / 90)
    goal_pts     = (hybrid_xG_90 * pos_mask_goal * finishing_mult) * (actual_mins / 90)
    save_pts     = (df['saves_per_90'] * df['fixture_defence_multiplier'] * save_mask) * (actual_mins / 90)
    cards_pts    = (
        (red_card_mask * df['red_cards_index_per_90'])
        + (yellow_card_mask * df['yellow_cards_per_90'])
    ) * (actual_mins / 90)

    estimate_bps_calibrate = (
        minutes_pts * minutes_mask_bonus
        + (hybrid_xG_90 * pos_mask_goal_bonus * finishing_mult) * (actual_mins / 90)
        + (hybrid_xA_90 * assist_mask_bonus) * (actual_mins / 90)
        + (pos_mask_cs_bonus * cs_prob_actual * (actual_mins >= 60).astype(float))
        + (df['saves_per_90'] * df['fixture_defence_multiplier'] * save_mask_bonus) * (actual_mins / 90)
        + (penalty_mask_bonus * adj_xGC_actual)
        + (defcon_mask_bonus * defcon_prob_calibrate)
        + (red_card_mask_bonus  * df['red_cards_index_per_90']) * (actual_mins / 90)
        + (yellow_card_mask_bonus * df['yellow_cards_per_90'])  * (actual_mins / 90)
    )

    # Fit internal calibrators if none were provided externally.
    # Requires ≥50 rows with actual minutes ≥45 for statistical reliability.
    if bps_calibrators is None:
        has_actual = actual_mins >= 45
        if has_actual.sum() >= 50:
            internal_df = pd.DataFrame({
                'position':               df.loc[has_actual, 'position'],
                'estimate_bps_calibrate': estimate_bps_calibrate[has_actual],
                'actual_avg_bps':         (df['bps_per_90'] * (actual_mins / 90))[has_actual],
                'actual_minutes':         actual_mins[has_actual],
            })
            bps_calibrators = fit_bps_calibration(
                internal_df,
                estimate_col='estimate_bps_calibrate',
                actual_col='actual_avg_bps',
                min_minutes=45,
            )

    # Apply calibrators to the forward-looking BPS estimate (estimate_bps).
    # If calibrators exist: 90% calibrated model + 10% historical BPS per 90 (regulariser).
    # If no calibrators: 50/50 blend with historical BPS (more uncertainty, equal weight).
    if bps_calibrators:
        calibrated  = _apply_bps_calibration(estimate_bps, df['position'], bps_calibrators)
        historical  = (df['bps_per_90'] * (df['minutes_IDX'] / 90)).clip(lower=0)
        hybrid_bps_abs = 0.90 * calibrated + 0.10 * historical
    else:
        historical     = (df['bps_per_90'] * (df['minutes_IDX'] / 90)).clip(lower=0)
        hybrid_bps_abs = 0.50 * estimate_bps + 0.50 * historical

    df['hybrid_bps_abs'] = hybrid_bps_abs

    # --- Bonus point expected value from classifier ---
    # predict_proba returns [P(0), P(1), P(2), P(3)] for each player.
    # Dot product with class values gives E[bonus points].
    probs      = bonus_model.predict_proba(hybrid_bps_abs.values.reshape(-1, 1))
    classes    = bonus_model.classes_.astype(float)
    ev_bonus   = probs @ classes
    bonus_rate = np.clip(ev_bonus, 0, 3)

    df['bonus_component'] = pd.Series(bonus_rate, index=df.index)

    # --- C10. Perf_IDX: sum of all expected components ---
    # This is the core expected points score. It is in regular FPL points space.
    # All downstream blend operations must stay in this same space.
    df['Perf_IDX'] = (
        df['minutes_component']
        + df['CLEAN_SHEET_INDEX']
        + df['CONCEDED_PENALTY']
        + df['defcon_component']
        + df['ASSIST_INDEX']
        + df['GOAL_INDEX']
        + df['bonus_component']
        + df['RED_YELLOW_PENALTY']
        + df['SAVE_component']
    )

    # =========================================================================
    # PART E — FIXTURE CALIBRATED POINTS (model validation only)
    # Re-runs the full model on actual_minutes instead of minutes_IDX.
    # Used to compare model predictions vs real outcomes for calibration checks.
    # NOT used in the solver — do not use as an input to custom_score.
    # =========================================================================

    if bps_calibrators:
        calibrated_calibrate     = _apply_bps_calibration(
            estimate_bps_calibrate, df['position'], bps_calibrators
        )
        historical_calibrate     = (df['bps_per_90'] * (actual_mins / 90)).clip(lower=0)
        hybrid_bps_abs_calibrate = 0.90 * calibrated_calibrate + 0.10 * historical_calibrate
    else:
        historical_calibrate     = (df['bps_per_90'] * (actual_mins / 90)).clip(lower=0)
        hybrid_bps_abs_calibrate = 0.50 * estimate_bps_calibrate + 0.50 * historical_calibrate

    df['estimate_bps_calibrate'] = estimate_bps_calibrate
    df['actual_avg_bps']         = df['bps_per_90'] * (actual_mins / 90)

    probs_calibrate    = bonus_model.predict_proba(hybrid_bps_abs_calibrate.values.reshape(-1, 1))
    ev_bonus_calibrate = probs_calibrate @ bonus_model.classes_.astype(float)
    bonus_pts_calibrate = pd.Series(
        np.clip(ev_bonus_calibrate, 0, 3), index=df.index
    )

    df['fixture_calibrated_points'] = (
        minutes_pts
        + cs_pts
        + deduction_pts
        + defcon_pts
        + assist_pts
        + goal_pts
        + bonus_pts_calibrate
        + cards_pts
        + save_pts
    )

    # =========================================================================
    # PART F — CEILING SCORE (Statistical Upside via Variance Aggregation)
    #
    # WHAT CHANGED FROM THE ORIGINAL:
    #   Old approach:
    #     ceiling_goals   = exp_goals   + √exp_goals    ← 1-std ceiling on goals alone
    #     ceiling_assists = exp_assists + √exp_assists  ← 1-std ceiling on assists alone
    #     ceiling_score   = Σ(ceiling_X for each X)    ← INCORRECT: assumes all
    #                                                      components peak simultaneously
    #
    #   New approach (mathematically correct):
    #     Compute Var[points] per component independently.
    #     Sum all variances: Var[total] = Σ Var[component_i]   (components are independent)
    #     Take ONE square root: std[total] = √Var[total]
    #     ceiling_score = Perf_IDX + UPSIDE_Z * std[total]
    #
    #   Why this is correct:
    #     Haaland cannot score 2 goals AND earn a clean sheet AND get maximum saves
    #     all at their individual ceiling rates in the same match. By summing variances
    #     first, we correctly account for the fact that joint extreme events are far
    #     less likely than individual extreme events. The result is a realistic upside
    #     scenario rather than a theoretical all-components-maximum.
    #
    #   Result:
    #     Players with genuinely high scoring variance (e.g. boom-bust FWDs) still
    #     get wide ceilings. Consistent low-variance DEFs get tight ceilings.
    #     No hardcoded co-occurrence discount needed — the math handles it.
    # =========================================================================

    # --- F1. Component expected values (re-use where already computed, recalculate where needed) ---
    exp_goals   = hybrid_xG_90 * (df['minutes_IDX'] / 90)
    exp_assists = hybrid_xA_90 * (df['minutes_IDX'] / 90)
    exp_saves   = (
        df['saves_per_90'] * df['fixture_defence_multiplier'] * save_mask
    ) * (df['minutes_IDX'] / 90)

    # --- F2. Variance: Goal Points ---
    # Goals ~ Poisson(λ). For Poisson: Var[X] = λ.
    # Points from goals = goals × pts_per_goal, so:
    # Var[goal_pts] = Var[goals] × pts_per_goal² = λ × (pos_mask_goal × finishing_mult)²
    goals_pts_per_goal = pos_mask_goal * finishing_mult
    var_goal_pts       = exp_goals * (goals_pts_per_goal ** 2)

    # --- F3. Variance: Assist Points ---
    # Assists ~ Poisson(λ).
    # Var[assist_pts] = λ × assist_pts² = exp_assists × 9  (3 pts per assist, 3²=9)
    var_assist_pts = exp_assists * (assist_mask ** 2)

    # --- F4. Variance: Clean Sheet Points ---
    # CS ~ Bernoulli(p). Var[Bernoulli] = p(1-p).
    # cs_prob_pred is already computed in Part C.
    # Only GKP/DEF receive CS points (pos_mask_cs = 0 for MID/FWD).
    var_cs_pts = (
        (pos_mask_cs ** 2)
        * cs_prob_pred * (1 - cs_prob_pred)
        * (df['minutes_IDX'] >= 60).astype(float)
    )

    # --- F5. Variance: Save Points (GKP only) ---
    # Saves ~ Poisson(λ). Var[save_pts] = λ × save_pts_per_save²
    # save_mask = 1/3, so variance per save = (1/3)² = 1/9.
    var_save_pts = exp_saves * (save_mask ** 2)

    # --- F6. Variance: Bonus Points ---
    # Extract directly from the classifier's probability distribution.
    # Var[bonus] = E[bonus²] - E[bonus]²
    # This is the most accurate source — uses the full discrete probability
    # mass function rather than a normal approximation.
    bonus_classes = bonus_model.classes_.astype(float)
    e_bonus_pred  = probs @ bonus_classes              # already computed in Part D
    e_bonus_sq    = probs @ (bonus_classes ** 2)
    var_bonus_pts = pd.Series(
        (e_bonus_sq - e_bonus_pred ** 2).clip(0),
        index=df.index
    )

    # --- F7. Variance: Defensive Contribution Bonus ---
    # Defcon event ~ Bernoulli(p) approximation on threshold exceedance.
    # Var[Bernoulli] = p(1-p).
    # defcon_mask = 2 (2 pts per defcon bonus event).
    var_defcon_pts = (defcon_mask ** 2) * df['defcon_prob'] * (1 - df['defcon_prob'])

    # --- F8. Total Score Variance and Standard Deviation ---
    # Sum of independent component variances.
    # Independence assumption: goals, assists, CS, saves, bonus, defcon
    # are approximately independent conditional on fixture and minutes.
    # (A player scoring a goal doesn't prevent them from also earning saves —
    # but the JOINT extreme of all peaking simultaneously is penalised by the
    # √(ΣVar) formula vs the old Σ(√Var_i) formula.)
    total_variance = (
        var_goal_pts
        + var_assist_pts
        + var_cs_pts
        + var_save_pts
        + var_bonus_pts
        + var_defcon_pts
    ).clip(lower=0)

    total_std = np.sqrt(total_variance)
    df['score_std'] = total_std   # store for diagnostics and calibration checks

    # --- F9. Ceiling Score ---
    # ceiling_score = Perf_IDX + Z * total_std
    #
    # UPSIDE_Z=1.5 targets the 93rd percentile of the total score distribution,
    # assuming approximate normality of the sum (Central Limit Theorem).
    # This is a "good game" scenario, not a theoretical maximum.
    #
    # UNIT INTEGRITY:
    # ceiling_score is in the same regular FPL points space as Perf_IDX.
    # Blending them in create_optimized_custom_score is unit-safe.
    # The old "captain_score" name was misleading — this is NOT doubled points.
    # Captain selection in the solver uses a separate captain_column (Perf_IDX × 2).
    df['ceiling_score'] = df['Perf_IDX'] + UPSIDE_Z * total_std

    return df


# --- CELL 36 ---
def create_optimized_custom_score(
    df: pd.DataFrame,
    differential_weight: float,
    upside_weight: float,
    perf_col: str = 'Perf_IDX',
    upside_metric: str = 'ceiling_score',
    s_curve_midpoint: float = 0.18,
    s_curve_steepness: float = 18.0,
    prior_weight: float = 10.0,
    visualize: bool = False,
) -> pd.DataFrame:
    """
    Produces custom_score and raw_bonus_multiplier for every player row.

    Outputs
    -------
    custom_score         : squad selection score (honest regular-points space blend)
    raw_bonus_multiplier : transfer cost discount [0, 0.15] — consumed by solver only,
                           never added to custom_score.

    Parameters
    ----------
    df                  : projection dataframe (must contain perf_col and upside_metric)
    differential_weight : strength of ownership-based transfer cost discount (0 = off)
    upside_weight       : base blend weight toward ceiling_score (clipped to [0, 0.15])
    perf_col            : expected points column ('Perf_IDX')
    upside_metric       : upside ceiling column ('ceiling_score')
    s_curve_midpoint    : ownership fraction at which differential bonus = 50% of max
    s_curve_steepness   : sharpness of the S-curve transition
    prior_weight        : Bayesian smoothing strength for per-position CV estimates
    visualize           : if True, renders the 6-panel diagnostic dashboard
    """

    df = df.copy()

    # [PATCH 1A]: Add safe default to prevent KeyError when differential_weight=0
    df['raw_bonus_multiplier'] = 0.0

    # --- Safe defaults if columns are missing ---
    if perf_col not in df.columns:
        df[perf_col] = 0.0
    if upside_metric not in df.columns:
        df[upside_metric] = 0.0

    # --- Always initialise — solver reads this even when differential_weight=0 ---
    df['raw_bonus_multiplier'] = 0.0

    def _safe_col(col, default=0.0):
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)

    upside_weight_clipped = np.clip(upside_weight, 0.0, 0.15)

    # =========================================================================
    # STEP 1 — ACTIVE PLAYER MASK
    # Rank and normalise only among players projected to play.
    # Prefer forward-looking minutes_IDX; fall back to actual_minutes or all rows.
    # =========================================================================
    if 'minutes_IDX' in df.columns and (df['minutes_IDX'] > 0).any():
        active_mask = df['minutes_IDX'] > 0
    elif 'actual_minutes' in df.columns and (df['actual_minutes'] > 0).any():
        active_mask = df['actual_minutes'] > 0
    else:
        active_mask = pd.Series(True, index=df.index)

    # =========================================================================
    # STEP 2 — CEILING GAP AND PERCENTILE RANK
    # ceiling_gap : how many more points does the player's ceiling project vs expected?
    # gap_ratio   : ceiling gap relative to the player's own expected score (normalised).
    # gap_pct     : percentile rank of gap_ratio among active players [0, 1].
    # =========================================================================
    df['ceiling_gap'] = (df[upside_metric] - df[perf_col]).clip(lower=0)
    df['gap_ratio']   = df['ceiling_gap'] / (df[perf_col] + 1e-9)

    df['gap_pct'] = 0.0
    if active_mask.any():
        df.loc[active_mask, 'gap_pct'] = (
            df.loc[active_mask, 'gap_ratio'].rank(pct=True)
        )

    # =========================================================================
    # STEP 3 — DYNAMIC UPSIDE WEIGHT
    # Per-player blend weight toward ceiling_score.
    # Players with a larger relative ceiling gap get a higher weight (up to 0.30).
    # Formula: base × (0.75 + 0.50 × gap_pct)
    #   gap_pct=0 → weight = base × 0.75  (lowest ceiling gap, still some upside)
    #   gap_pct=1 → weight = base × 1.25  (highest ceiling gap, boosted)
    # =========================================================================
    df['dynamic_upside'] = (
        upside_weight_clipped * (0.75 + 0.50 * df['gap_pct'])
    ).clip(lower=0.0, upper=0.30)

    # =========================================================================
    # STEP 4 — CUSTOM SCORE
    # Symmetric blend: same formula for every player, no asymmetric inflation.
    # Both perf_col and upside_metric are in regular FPL points space so the
    # blend output is also in regular points space — transfer math stays clean.
    # =========================================================================
    df['custom_score'] = (
        (1 - df['dynamic_upside']) * df[perf_col]
        + df['dynamic_upside']     * df[upside_metric]
    )

    # =========================================================================
    # STEP 5 — DIFFERENTIAL QUALITY SIGNAL
    # A [0, 1] normalised signal capturing how good this player is as a
    # differential pick — combining performance quality, upside room, and
    # starting reliability. NOT in points space.
    # Multiplicative dampener: players unlikely to start collapse toward 0
    # regardless of how good they look on paper.
    # =========================================================================
    start_per_gp      = _safe_col('start_per_gameplayed').clip(0, 1)
    consec_streak     = _safe_col('consecutive_start_streak').clip(0, 5) / 5
    start_reliability = (start_per_gp * consec_streak).fillna(0.0)

    perf_max = df.loc[active_mask, perf_col].max()      if active_mask.any() else 1.0
    gap_max  = df.loc[active_mask, 'ceiling_gap'].max() if active_mask.any() else 1.0
    perf_max = perf_max if pd.notna(perf_max) and perf_max > 0 else 1.0
    gap_max  = gap_max  if pd.notna(gap_max)  and gap_max  > 0 else 1.0

    df['diff_quality_signal'] = (
        0.60 * df[perf_col]      / (perf_max + 1e-9)
        + 0.25 * df['ceiling_gap'] / (gap_max  + 1e-9)
        + 0.15 * start_reliability
    )
    df['diff_quality_signal'] = (
        df['diff_quality_signal'].fillna(0.0) * start_reliability.clip(0, 1)
    ).fillna(0.0)

    # =========================================================================
    # STEP 6 — DIFFERENTIAL BONUS MULTIPLIER
    # Combines:
    #   (A) unowned_potential — S-curve on ownership: low-owned → high potential
    #   (B) dynamic_diff_weight — per-position Bayesian-smoothed quality scaling
    # Result clipped to [0, 0.15]: maximum 15% discount on a -4 transfer hit.
    # Stored in raw_bonus_multiplier — consumed by the solver's transfer cost
    # calculation, NEVER added to custom_score.
    # =========================================================================
    diff_multiplier_map = {}
    pos_cv              = None

    if differential_weight > 0 and 'selected_by_percent' in df.columns:

        df['selected_by_percent'] = pd.to_numeric(
            df['selected_by_percent'], errors='coerce'
        ).fillna(0)

        ownership = df['selected_by_percent'] / 100.0

        # S-curve maps ownership to differential opportunity [0, 1].
        # Below midpoint → bonus approaches 1. Above midpoint → bonus approaches 0.
        unowned_potential = 1 / (
            1 + np.exp(s_curve_steepness * (ownership - s_curve_midpoint))
        )

        if 'position' in df.columns and active_mask.any():

            # Per-position Bayesian smoothing of diff_quality_signal.
            # Pulls small-sample positions (GKP, FWD) toward the global mean
            # to prevent extreme CV estimates from producing wild multipliers.
            active_df        = df.loc[active_mask]
            diff_global_mean = active_df['diff_quality_signal'].mean()
            diff_global_std  = max(active_df['diff_quality_signal'].std(), 1e-9)

            pos_counts   = active_df.groupby('position').size()
            pos_mean_raw = active_df.groupby('position')['diff_quality_signal'].mean()
            pos_std_raw  = (
                active_df.groupby('position')['diff_quality_signal']
                .std().fillna(diff_global_std)
            )

            smoothed_mean = (
                (pos_counts * pos_mean_raw) + (prior_weight * diff_global_mean)
            ) / (pos_counts + prior_weight)
            smoothed_std  = (
                (pos_counts * pos_std_raw)  + (prior_weight * diff_global_std)
            ) / (pos_counts + prior_weight)

            # CV (coefficient of variation) measures spread relative to mean.
            # High-CV position = more quality variation → differential picks matter more.
            pos_cv = (smoothed_std / smoothed_mean.replace(0, np.nan)).fillna(1.0)
            diff_global_cv      = diff_global_std / diff_global_mean if diff_global_mean > 0 else 1.0
            diff_multiplier_map = (
                pos_cv / diff_global_cv
            ).clip(lower=0.6, upper=1.4).to_dict()

            dynamic_diff_weight = (
                df['position'].map(diff_multiplier_map).fillna(1.0)
                * differential_weight
            )
        else:
            dynamic_diff_weight = differential_weight

        raw_bonus_multiplier       = dynamic_diff_weight * unowned_potential
        df['raw_bonus_multiplier'] = np.clip(raw_bonus_multiplier, 0.0, 0.15)

        df['captain_idx'] = (
            0.3 * df[perf_col]
            + 0.7 * df[upside_metric]
        )

    # =========================================================================
    # STEP 7 — VISUALIZATION
    # =========================================================================
    if visualize and active_mask.any():

        active_df = df.loc[active_mask].copy()

        # Build the mapping table shown in Panel 5
        mapping_data = {}

        if 'position' in active_df.columns:
            mapping_data['Avg_Perf_IDX']      = active_df.groupby('position')[perf_col].mean().round(2)
            mapping_data['Avg_Custom_Score']   = active_df.groupby('position')['custom_score'].mean().round(2)
            mapping_data['Avg_Gap_Ratio']      = active_df.groupby('position')['gap_ratio'].mean().round(3)
            mapping_data['Avg_Dynamic_Upside'] = active_df.groupby('position')['dynamic_upside'].mean().round(3)

            if 'score_std' in active_df.columns:
                mapping_data['Avg_Score_Std'] = active_df.groupby('position')['score_std'].mean().round(3)

            if differential_weight > 0 and pos_cv is not None:
                mapping_data['Smoothed_CV']       = pos_cv.round(3)
                mapping_data['Diff_Final_Weight']  = (
                    pd.Series(diff_multiplier_map) * differential_weight
                ).round(3)

        mapping_table = pd.DataFrame(mapping_data) if mapping_data else pd.DataFrame()

        print('\n=== DYNAMIC MAPPING SUMMARY TABLE ===')
        if not mapping_table.empty:
            display(mapping_table)

        _render_sync_plots(
            active_df        = active_df,
            mapping_table    = mapping_table,
            upside_metric    = upside_metric,
            differential_weight    = differential_weight,
            upside_weight_clipped  = upside_weight_clipped,
        )

    return df

# --- CELL 37 ---
