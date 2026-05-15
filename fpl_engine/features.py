import pandas as pd
import numpy as np
from datetime import datetime
import optuna
from tqdm.auto import tqdm
from .data import get_current_players_df, get_fixture_df, get_team_df, enforce_datatypes

def compute_rolling_team_ratings(
    raw_history_df: pd.DataFrame,
    ema_alpha: float = 0.15,
    min_fixtures: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes rolling EMA team attack (xG) and defence (xGC) ratings
    from aggregated player-level fixture history.

    Returns
    -------
    team_fixture_df : historical fixture-level ratings (for past fixture merge)
    latest_ratings  : most recent rating per team (for future fixture fallback)
    """
    # Strictly deduplicate player-fixture rows to prevent double-counting xG
    hist = raw_history_df.drop_duplicates(subset=['id_player', 'id_fixture']).copy()

    # raw_history_df has no 'team' column — resolve from player data
    if 'team' not in hist.columns:
        player_team_map = (
            get_current_players_df()[['id', 'team']]
            .rename(columns={'id': 'id_player'})
        )
        hist = pd.merge(hist, player_team_map, on='id_player', how='left')

    # Use median of top-7 player minutes per fixture
    def _team_mins(x):
        return x.nlargest(7).median() if len(x) >= 3 else x.max()

    team_fixture = (
        hist.groupby(['team', 'id_fixture', 'kickoff_time', 'was_home'])
        .agg(
            team_xG  = ('expected_goals',          'sum'),
            team_xGC = ('expected_goals_conceded',  'sum'),
            team_mins= ('minutes',                  _team_mins),
        )
        .reset_index()
        .sort_values(['team', 'kickoff_time'])
    )

    # Convert to per-90 rates — clip minutes floor to avoid divide-by-zero
    team_fixture['team_mins'] = team_fixture['team_mins'].clip(lower=45)
    team_fixture['team_xG_per90']  = (team_fixture['team_xG']  / team_fixture['team_mins']) * 90
    team_fixture['team_xGC_per90'] = (team_fixture['team_xGC'] / team_fixture['team_mins']) * 90

    # Walk-forward EMA — shift(1) prevents data leakage into current fixture
    team_fixture['rolling_xG_per90'] = (
        team_fixture.groupby('team')['team_xG_per90']
        .transform(lambda x: x.shift(1).ewm(alpha=ema_alpha, adjust=False).mean())
    )
    team_fixture['rolling_xGC_per90'] = (
        team_fixture.groupby('team')['team_xGC_per90']
        .transform(lambda x: x.shift(1).ewm(alpha=ema_alpha, adjust=False).mean())
    )

    # Mask sparse early-season data — fallback handled in blend step
    fixture_count = team_fixture.groupby('team').cumcount()
    sparse_mask   = fixture_count < min_fixtures
    team_fixture.loc[sparse_mask, 'rolling_xG_per90']  = np.nan
    team_fixture.loc[sparse_mask, 'rolling_xGC_per90'] = np.nan

    # FIX 2: Extract most recent rating per team — used for future fixture fallback
    latest_ratings = (
        team_fixture
        .sort_values('kickoff_time')
        .groupby('team')[['rolling_xG_per90', 'rolling_xGC_per90']]
        .last()
        .reset_index()
        .rename(columns={
            'rolling_xG_per90' : 'latest_xG_per90',
            'rolling_xGC_per90': 'latest_xGC_per90',
        })
    )

    return (
        team_fixture[[
            'team', 'id_fixture', 'kickoff_time', 'was_home',
            'team_xG_per90', 'team_xGC_per90',
            'rolling_xG_per90', 'rolling_xGC_per90'
        ]],
        latest_ratings
    )

# --- CELL 21 ---
def blend_team_ratings(
    team_fixture_df: pd.DataFrame,
    latest_ratings: pd.DataFrame,
    fpl_team_df: pd.DataFrame,
    league_avg_xG: float  = 1.45,
    league_avg_xGC: float = 1.45,
    blend_alpha: float    = 0.75,
    min_fixtures_full_trust: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    df = team_fixture_df.copy()

    # [FIX 3]: Sort explicitly before cumcount to prevent backwards blend weights
    df = df.sort_values(['team', 'kickoff_time'])
    fpl_id_map = fpl_team_df.set_index('id')

    # [CRITICAL SYSTEM NOTE]: FPL API Naming Convention Inversion
    # _away = TRUE HOME STRENGTH | _home = TRUE AWAY STRENGTH
    df['fpl_attack_norm'] = np.where(
        df['was_home'],
        df['team'].map(fpl_id_map['strength_attack_away']),  # True Home
        df['team'].map(fpl_id_map['strength_attack_home'])   # True Away
    )
    df['fpl_defence_norm'] = np.where(
        df['was_home'],
        df['team'].map(fpl_id_map['strength_defence_away']), # True Home
        df['team'].map(fpl_id_map['strength_defence_home'])  # True Away
    )

    # [FIX 5]: Compute bounds dynamically from actual data to prevent inverted values
    all_strength_vals = pd.concat([
        fpl_team_df['strength_attack_home'],
        fpl_team_df['strength_attack_away'],
        fpl_team_df['strength_defence_home'],
        fpl_team_df['strength_defence_away'],
    ])
    fpl_att_min = all_strength_vals.min()
    fpl_att_max = max(all_strength_vals.max(), fpl_att_min + 1) # Fallback to avoid div zero

    # Normalize FPL ratings to xG space
    for col, avg in [('fpl_attack_norm', league_avg_xG), ('fpl_defence_norm', league_avg_xGC)]:
        df[col] = (
            df[col]
            .sub(fpl_att_min)
            .div(fpl_att_max - fpl_att_min)
            .mul(avg * 2)
            .add(avg * 0.5)
        )

    # Dynamic blend weight — scales from 0 to blend_alpha as fixtures accumulate
    fixture_count = df.groupby('team').cumcount()
    w = (fixture_count / min_fixtures_full_trust).clip(upper=1.0) * blend_alpha

    df['final_attack_rating'] = (
        w       * df['rolling_xG_per90'].fillna(league_avg_xG)
        + (1-w) * df['fpl_attack_norm']
    )
    df['final_defence_rating'] = (
        w       * df['rolling_xGC_per90'].fillna(league_avg_xGC)
        + (1-w) * df['fpl_defence_norm']
    )

    # [FIX 6]: Split Latest Ratings into Home/Away variants (preserving API inversion rule)
    latest = latest_ratings.copy()

    # True Home Strengths (Mapping to API's _away suffix)
    latest['fpl_attack_norm_home'] = latest['team'].map(fpl_id_map['strength_attack_away']).sub(fpl_att_min).div(fpl_att_max - fpl_att_min).mul(league_avg_xG * 2).add(league_avg_xG * 0.5)
    latest['fpl_defence_norm_home'] = latest['team'].map(fpl_id_map['strength_defence_away']).sub(fpl_att_min).div(fpl_att_max - fpl_att_min).mul(league_avg_xGC * 2).add(league_avg_xGC * 0.5)

    # True Away Strengths (Mapping to API's _home suffix)
    latest['fpl_attack_norm_away'] = latest['team'].map(fpl_id_map['strength_attack_home']).sub(fpl_att_min).div(fpl_att_max - fpl_att_min).mul(league_avg_xG * 2).add(league_avg_xG * 0.5)
    latest['fpl_defence_norm_away'] = latest['team'].map(fpl_id_map['strength_defence_home']).sub(fpl_att_min).div(fpl_att_max - fpl_att_min).mul(league_avg_xGC * 2).add(league_avg_xGC * 0.5)

    # Output distinct home and away latest ratings
    w_latest = blend_alpha
    latest['latest_final_attack_home']  = w_latest * latest['latest_xG_per90'].fillna(league_avg_xG)  + (1-w_latest) * latest['fpl_attack_norm_home']
    latest['latest_final_defence_home'] = w_latest * latest['latest_xGC_per90'].fillna(league_avg_xGC) + (1-w_latest) * latest['fpl_defence_norm_home']

    latest['latest_final_attack_away']  = w_latest * latest['latest_xG_per90'].fillna(league_avg_xG)  + (1-w_latest) * latest['fpl_attack_norm_away']
    latest['latest_final_defence_away'] = w_latest * latest['latest_xGC_per90'].fillna(league_avg_xGC) + (1-w_latest) * latest['fpl_defence_norm_away']

    return (
        df[['team', 'id_fixture', 'kickoff_time', 'was_home', 'final_attack_rating', 'final_defence_rating']],
        latest[['team', 'latest_final_attack_home', 'latest_final_defence_home', 'latest_final_attack_away', 'latest_final_defence_away']]
    )

# --- CELL 22 ---
def get_historical_normalized_stats(
    raw_history_df,
    team_df,
    params,
    global_dists,
    team_ratings_df=None,
    latest_team_ratings=None # [FIX 2]: Ensure this is passed to prevent silent NaNs
):
    history_df = raw_history_df.sort_values(['id_player', 'kickoff_time']).copy()

    alpha_att      = params.get('fixture_alpha_att', 0.15)
    alpha_def      = params.get('fixture_alpha_def', 0.15)
    recency_alpha  = params.get('recency_ema_alpha', 0.20)

    if 'team' not in history_df.columns:
        player_team_map = (
            get_current_players_df()[['id', 'team']]
            .rename(columns={'id': 'id_player'})
        )
        history_df = pd.merge(history_df, player_team_map, on='id_player', how='left')

    if team_ratings_df is not None:
        own_xg = team_ratings_df[['team', 'id_fixture', 'final_attack_rating', 'final_defence_rating']].add_prefix('own_')
        history_df = pd.merge(history_df, own_xg, left_on=['team', 'id_fixture'], right_on=['own_team', 'own_id_fixture'], how='left')

        opp_xg = team_ratings_df[['team', 'id_fixture', 'final_attack_rating', 'final_defence_rating']].add_prefix('opp_')
        history_df = pd.merge(history_df, opp_xg, left_on=['opponent', 'id_fixture'], right_on=['opp_team', 'opp_id_fixture'], how='left')

        # [FIX 2 & 6]: Fill unmatched historical rows with Home/Away split logic
        if latest_team_ratings is not None:
            latest_map = latest_team_ratings.set_index('team')

            # Own Team Fallback
            history_df['own_final_attack_rating'] = history_df['own_final_attack_rating'].fillna(
                pd.Series(np.where(history_df['was_home'], history_df['team'].map(latest_map['latest_final_attack_home']), history_df['team'].map(latest_map['latest_final_attack_away'])), index=history_df.index)
            )
            history_df['own_final_defence_rating'] = history_df['own_final_defence_rating'].fillna(
                pd.Series(np.where(history_df['was_home'], history_df['team'].map(latest_map['latest_final_defence_home']), history_df['team'].map(latest_map['latest_final_defence_away'])), index=history_df.index)
            )

            # Opp Team Fallback (~was_home because Opponent perspective is inverted)
            history_df['opp_final_attack_rating'] = history_df['opp_final_attack_rating'].fillna(
                pd.Series(np.where(~history_df['was_home'], history_df['opponent'].map(latest_map['latest_final_attack_home']), history_df['opponent'].map(latest_map['latest_final_attack_away'])), index=history_df.index)
            )
            history_df['opp_final_defence_rating'] = history_df['opp_final_defence_rating'].fillna(
                pd.Series(np.where(~history_df['was_home'], history_df['opponent'].map(latest_map['latest_final_defence_home']), history_df['opponent'].map(latest_map['latest_final_defence_away'])), index=history_df.index)
            )

        att_delta = history_df['own_final_attack_rating'] - history_df['opp_final_defence_rating']
        def_delta = history_df['opp_final_attack_rating'] - history_df['own_final_defence_rating']

    else:
        # [FIX 1]: Resolve Scope crash and cross-map inverted API data correctly
        strength_cols = [
            'id', 'strength_attack_home', 'strength_attack_away',
            'strength_defence_home', 'strength_defence_away'
        ]
        opp_df = team_df[strength_cols].add_prefix('opp_')
        own_df = team_df[strength_cols].add_prefix('own_')
        history_df = pd.merge(history_df, opp_df, left_on='opponent', right_on='opp_id', how='left')
        history_df = pd.merge(history_df, own_df, left_on='team', right_on='own_id', how='left')

        # Own team perspective cross-map
        team_attack_rating  = np.where(history_df['was_home'], history_df['own_strength_attack_away'], history_df['own_strength_attack_home'])
        team_defence_rating = np.where(history_df['was_home'], history_df['own_strength_defence_away'], history_df['own_strength_defence_home'])

        # Opponent team perspective cross-map
        opp_defence_rating  = np.where(~history_df['was_home'], history_df['opp_strength_defence_away'], history_df['opp_strength_defence_home'])
        opp_attack_rating   = np.where(~history_df['was_home'], history_df['opp_strength_attack_away'], history_df['opp_strength_attack_home'])

        att_delta = team_attack_rating - opp_defence_rating
        def_delta = opp_attack_rating  - team_defence_rating

    # Normalization Calculations
    att_mu, att_std = global_dists['att_mu'], global_dists['att_std']
    def_mu, def_std = global_dists['def_mu'], global_dists['def_std']

    raw_att_multiplier = 1 + (alpha_att * ((att_delta - att_mu) / att_std))
    raw_def_multiplier = 1 + (alpha_def * ((def_delta - def_mu) / def_std))

    Z_CLIP = 3.0
    history_df['hist_attack_multiplier']  = np.clip(raw_att_multiplier, max(0.5, 1.0 - alpha_att * Z_CLIP), min(2.0, 1.0 + alpha_att * Z_CLIP))
    history_df['hist_defence_multiplier'] = np.clip(raw_def_multiplier, max(0.5, 1.0 - alpha_def * Z_CLIP), min(2.0, 1.0 + alpha_def * Z_CLIP))

    # --- Normalised per-game totals ---
    history_df['norm_xG']            = history_df['expected_goals']           / history_df['hist_attack_multiplier']
    history_df['norm_xA']            = history_df['expected_assists']          / history_df['hist_attack_multiplier']
    history_df['norm_xGC']           = history_df['expected_goals_conceded']   / history_df['hist_defence_multiplier']
    history_df['norm_threat']        = history_df['threat']                    / history_df['hist_attack_multiplier']
    history_df['norm_creativity']    = history_df['creativity']                / history_df['hist_attack_multiplier']
    history_df['norm_goals_scored']  = history_df['goals_scored']              / history_df['hist_attack_multiplier']
    history_df['norm_goals_conceded']= history_df['goals_conceded']            / history_df['hist_defence_multiplier']
    history_df['norm_saves']         = history_df['saves']                     / history_df['hist_defence_multiplier']
    history_df['norm_defcon']        = history_df['defensive_contribution']    / history_df['hist_defence_multiplier']

    played_mask = history_df['minutes'] > 0
    history_df['norm_xGC_per_90'] = 0.0
    minutes_floored = history_df.loc[played_mask, 'minutes'].clip(lower=15.0)
    history_df.loc[played_mask, 'norm_xGC_per_90'] = (history_df.loc[played_mask, 'norm_xGC'] / (minutes_floored / 90.0))
    history_df['norm_clean_sheets'] = np.exp(-history_df['norm_xGC_per_90'])

    rate_cols = ['norm_xG', 'norm_xA', 'norm_xGC', 'norm_threat', 'norm_creativity', 'norm_goals_scored', 'norm_goals_conceded', 'norm_saves','norm_defcon']

    for col in rate_cols:
        r90_col = f'{col}_r90'
        history_df[r90_col] = 0.0
        history_df.loc[played_mask, r90_col] = (history_df.loc[played_mask, col] / (history_df.loc[played_mask, 'minutes'] / 90.0))

    history_df['norm_clean_sheets_r90'] = history_df['norm_clean_sheets']

    if recency_alpha == 0.0:
        result = (history_df.groupby('id_player')[['minutes'] + rate_cols + ['norm_clean_sheets']].sum().reset_index())
    else:
        grouped = history_df.groupby('id_player')
        totals = grouped['minutes'].agg(['sum', 'count']).rename(columns={'sum': 'minutes', 'count': 'games'})
        ema_cols = [f'{col}_r90' for col in rate_cols] + ['norm_clean_sheets_r90']
        emas = grouped[ema_cols].apply(lambda x: x.ewm(alpha=recency_alpha, adjust=False).mean().iloc[-1])
        result = pd.concat([totals, emas], axis=1).reset_index()

        for col in rate_cols:
            result[col] = result[f'{col}_r90'] * (result['minutes'] / 90.0)
        result['norm_clean_sheets'] = result['norm_clean_sheets_r90'] * result['games']

    keep_cols = ['id_player', 'minutes', 'norm_xG', 'norm_xA', 'norm_xGC', 'norm_threat', 'norm_creativity', 'norm_goals_scored', 'norm_goals_conceded', 'norm_saves', 'norm_clean_sheets', 'norm_defcon']
    return result[keep_cols]

# --- CELL 23 ---
def apply_normalized_baselines(df, season_norm_df):
    df = df.copy()

    # Merge normalized stats with a suffix to avoid clashing with existing columns
    df = pd.merge(df, season_norm_df, on='id_player', how='left', suffixes=('', '_norm'))

    # Helper to safely get the original column or derive it if possible
    def safe_fillna(norm_col, orig_col, per_90_col=None):
        if orig_col in df.columns:
            return df[norm_col].fillna(df[orig_col])
        elif per_90_col in df.columns and 'minutes' in df.columns:
            # Back-calculate raw from per_90 if raw is missing
            derived_raw = (df[per_90_col] * df['minutes']) / 90.0
            return df[norm_col].fillna(derived_raw)
        else:
            return df[norm_col].fillna(0)  # Ultimate fallback

    # Overwrite the base columns safely
    df['expected_goals'] = safe_fillna('norm_xG', 'expected_goals', 'expected_goals_per_90')
    df['expected_assists'] = safe_fillna('norm_xA', 'expected_assists', 'expected_assists_per_90')
    df['expected_goals_conceded'] = safe_fillna('norm_xGC', 'expected_goals_conceded', 'expected_goals_conceded_per_90')
    df['threat'] = safe_fillna('norm_threat', 'threat', 'threat_per_90')
    df['creativity'] = safe_fillna('norm_creativity', 'creativity', 'creativity_per_90')
    df['goals_scored'] = safe_fillna('norm_goals_scored', 'goals_scored')
    df['goals_conceded'] = safe_fillna('norm_goals_conceded', 'goals_conceded')
    df['saves'] = safe_fillna('norm_saves', 'saves', 'saves_per_90')
    df['defensive_contribution'] = safe_fillna('norm_defcon', 'defensive_contribution', 'defensive_contribution_per_90')

    # FIX Issue 3: norm_clean_sheets now carries a Poisson CS probability (0-1 range),
    # not a fractional count. Treat it as expected CS per game — do not raw-fill from
    # integer clean_sheets, which would break the probability scale.
    if 'norm_clean_sheets' in df.columns:
        df['clean_sheets'] = df['norm_clean_sheets'].fillna(0)
    elif 'clean_sheets' in df.columns:
        df['clean_sheets'] = df['clean_sheets'].fillna(0)
    else:
        df['clean_sheets'] = 0

    # FIX Issue 4: Explicit if/else — avoids Python truthiness bug where
    # `df['minutes'] if 'minutes' in df.columns else 1` always evaluates the
    # Series branch (non-empty Series is always truthy), making `else 1` unreachable.
    if 'minutes' in df.columns:
        df['minutes'] = df['minutes_norm'].fillna(df['minutes'])
    else:
        df['minutes'] = df['minutes_norm'].fillna(1)

    mins = df['minutes'].clip(lower=1)

    # Recalculate Per 90 Rates based on normalized totals
    df['expected_goals_per_90'] = (df['expected_goals'] / mins) * 90
    df['expected_assists_per_90'] = (df['expected_assists'] / mins) * 90
    df['expected_goals_conceded_per_90'] = (df['expected_goals_conceded'] / mins) * 90
    df['threat_per_90'] = (df['threat'] / mins) * 90
    df['creativity_per_90'] = (df['creativity'] / mins) * 90
    df['saves_per_90'] = (df['saves'] / mins) * 90
    df['defensive_contribution_per_90'] = (df['defensive_contribution'] / mins) * 90

    # Recalculate other metrics if minutes changed slightly
    if 'bps' in df.columns:
        df['bps_per_90'] = (df['bps'] / mins) * 90
    if 'defensive_contribution' in df.columns:
        df['defensive_contribution_per_90'] = (df['defensive_contribution'] / mins) * 90

    # Clean up intermediate columns
    drop_cols = [c for c in df.columns if c.startswith('norm_') or c.endswith('_norm')]
    df.drop(columns=drop_cols, inplace=True, errors='ignore')

    return df

# --- CELL 24 ---
def compute_global_z_distributions(team_ratings_df: pd.DataFrame) -> dict:
    """
    Calculates the league-wide attack and defense delta distributions
    in xG rating space instead of old FPL strength rating space.

    Accepts the output of blend_team_ratings (team_ratings_df).
    Uses the most recent rating per team to represent the current season state.
    """
    # Use most recent fixture rating per team (end-of-season snapshot)
    latest = (
        team_ratings_df
        .sort_values('kickoff_time', ascending=False)
        .drop_duplicates('team')
        [['team', 'final_attack_rating', 'final_defence_rating']]
    )

    all_matchups = pd.merge(
        latest.assign(key=1),
        latest.assign(key=1),
        on='key',
        suffixes=('_own', '_opp')
    ).query('team_own != team_opp')

    att_deltas = (
        all_matchups['final_attack_rating_own']
        - all_matchups['final_defence_rating_opp']
    )
    def_deltas = (
        all_matchups['final_attack_rating_opp']
        - all_matchups['final_defence_rating_own']
    )

    return {
        'att_mu':  float(att_deltas.mean()),
        'att_std': float(max(att_deltas.std(), 1e-6)),
        'def_mu':  float(def_deltas.mean()),
        'def_std': float(max(def_deltas.std(), 1e-6)),
    }

# --- CELL 25 ---
def compute_walkforward_minutes_features(
    fixture_player_df: pd.DataFrame,
    raw_history_df: pd.DataFrame,
    ema_alpha: float = 0.40,
) -> pd.DataFrame:

    # 1. Isolate the IDs of fixtures that have actually finished
    finished_fixtures = fixture_player_df[fixture_player_df['finished'] == True]['id_fixture'].unique()

    # 2. Filter raw history: drop duplicates, nulls, AND unplayed games
    hist = (
        raw_history_df
        .drop_duplicates(subset=['id_player', 'id_fixture'])
        .dropna(subset=['id_fixture', 'kickoff_time'])
    )
    hist = hist[hist['id_fixture'].isin(finished_fixtures)]
    hist = hist.sort_values(['id_player', 'kickoff_time']).copy()
    hist['minutes'] = hist['minutes'].fillna(0)

    records = []

    for player_id, grp in hist.groupby('id_player'):
        grp      = grp.reset_index(drop=True)
        mins_arr = grp['minutes'].values
        fix_arr  = grp['id_fixture'].values
        gw_arr   = grp['gameweek'].values
        n        = len(grp)

        ema = None

        for i in range(n + 1):
            if i == 0:
                rec = dict(
                    id_player=player_id,
                    _ref_fixture=fix_arr[0] if n > 0 else -1,
                    _is_future=(n == 0),
                    recent_minutes_form=0.0,
                    last_match_minutes=0.0,
                    consecutive_start_streak=0,
                    minutes_volatility=30.0,
                    minutes_trend_slope=0.0,
                )
                records.append(rec)
                if n > 0:
                    ema = float(mins_arr[0])
                continue

            prev_min = float(mins_arr[i - 1])
            ema      = prev_min if ema is None else ema_alpha * prev_min + (1 - ema_alpha) * ema

            streak = 0
            for j in range(i - 1, -1, -1):
                if mins_arr[j] >= 60: streak += 1
                else: break

            tail_idx = max(0, i - 6)
            tail6 = mins_arr[tail_idx : i]
            vol   = float(np.std(tail6, ddof=1)) if len(tail6) >= 3 else 30.0

            tail5_idx = max(0, i - 5)
            tail5_mins = mins_arr[tail5_idx : i]
            tail5_gws = gw_arr[tail5_idx : i]

            if len(tail5_mins) >= 3:
                slope = float(np.polyfit(tail5_gws, tail5_mins, 1)[0])
                slope = float(np.clip(slope, -20, 20))
            else:
                slope = 0.0

            ref_fix = fix_arr[i] if i < n else -1

            rec = dict(
                id_player=player_id,
                _ref_fixture=ref_fix,
                _is_future=(i >= n),
                recent_minutes_form=round(ema, 4),
                last_match_minutes=prev_min,
                consecutive_start_streak=streak,
                minutes_volatility=round(vol, 4),
                minutes_trend_slope=round(slope, 4),
            )
            records.append(rec)

    features_df = pd.DataFrame(records)

    future_features = features_df[features_df['_is_future']].drop(columns=['_ref_fixture', '_is_future'])
    past_features = features_df[~features_df['_is_future']].rename(columns={'_ref_fixture': 'id_fixture'}).drop(columns=['_is_future'])

    result = fixture_player_df.copy()

    result = pd.merge(result, past_features, on=['id_player', 'id_fixture'], how='left')

    unmatched = result['recent_minutes_form'].isna()
    if unmatched.any():
        result = pd.merge(result, future_features, on='id_player', how='left', suffixes=('', '_future'))
        for col in ['recent_minutes_form', 'last_match_minutes', 'consecutive_start_streak', 'minutes_volatility', 'minutes_trend_slope']:
            result[col] = result[col].fillna(result.get(f'{col}_future', np.nan))
        result.drop(columns=[c for c in result.columns if c.endswith('_future')], inplace=True)

    result.fillna({
        'recent_minutes_form': 0.0,
        'last_match_minutes': 0.0,
        'consecutive_start_streak': 0.0,
        'minutes_volatility': 30.0,
        'minutes_trend_slope': 0.0
    }, inplace=True)

    return result

# --- CELL 27 ---
def get_fixture_players_stats_df(
    params,
    raw_history_df,
    global_dists,
    team_ratings_df=None,
    latest_team_ratings=None,
):
    # 1. LOAD DATA
    player_df = enforce_datatypes(get_current_players_df())
    fixture_df = enforce_datatypes(get_fixture_df())

    # 2. MERGE FIXTURES AND PLAYERS
    fixture_player_df = pd.merge(fixture_df, player_df, on='team', how='left', suffixes=['_fixture', '_player'])
    team_df = get_team_df()

    # 3. APPLY NORMALIZED BASELINES EARLY
    # [FIX 2]: Ensure latest_team_ratings is sent down to prevent silent NaN cascade
    normalized_history = get_historical_normalized_stats(
        raw_history_df,
        team_df,
        params,
        global_dists,
        team_ratings_df=team_ratings_df,
        latest_team_ratings=latest_team_ratings
    )
    fixture_player_df = apply_normalized_baselines(fixture_player_df, normalized_history)

    # 4. MERGE FIXTURE-LEVEL ACTUALS (Deduplicated)
    historical_actuals = raw_history_df.drop_duplicates(subset=['id_player', 'id_fixture']).copy()
    historical_actuals = historical_actuals[['id_player', 'id_fixture', 'minutes', 'actual_points']]
    historical_actuals.rename(columns={'minutes': 'actual_minutes'}, inplace=True)

    fixture_player_df = pd.merge(fixture_player_df, historical_actuals, on=['id_player', 'id_fixture'], how='left')
    fixture_player_df['actual_minutes'] = fixture_player_df['actual_minutes'].fillna(0)
    fixture_player_df['actual_points'] = fixture_player_df['actual_points'].fillna(0)

    # 5. CALCULATE UNIFIED BASE TEAM STATS
    active_players = player_df.loc[player_df['minutes'] > 0].copy()
    team_stats = active_players.groupby('team')[['goals_scored', 'expected_goals', 'goals_conceded', 'expected_goals_conceded', 'minutes']].sum().reset_index()

    team_stats['team_expected_goals_per_90']          = (team_stats['expected_goals']          / team_stats['minutes']) * 90
    team_stats['team_expected_goals_conceded_per_90'] = (team_stats['expected_goals_conceded'] / team_stats['minutes']) * 90

    # [FIX 4]: Normalize team_xGC_per90 using the team's average defence multiplier to avoid systematic biases
    if team_ratings_df is not None:
        team_avg_def_mult = team_ratings_df.groupby('team')['final_defence_rating'].mean().reset_index()
        team_avg_def_mult.rename(columns={'final_defence_rating': 'team_avg_def_mult'}, inplace=True)
        team_stats = pd.merge(team_stats, team_avg_def_mult, on='team', how='left')
        team_stats['team_expected_goals_conceded_per_90'] = (
            team_stats['team_expected_goals_conceded_per_90'] / team_stats['team_avg_def_mult'].fillna(1.0)
        )

    fixture_player_df = pd.merge(
        fixture_player_df,
        team_stats[['team', 'team_expected_goals_per_90', 'team_expected_goals_conceded_per_90']],
        on='team', how='left'
    )

    # 6. GENERATE NORMALIZED DELTA MULTIPLIERS (xG-based, with FPL fallback)
    fixture_alpha_att = params.get('fixture_alpha_att', 0.14)
    fixture_alpha_def = params.get('fixture_alpha_def', 0.075)
    att_mu, att_std   = global_dists['att_mu'], global_dists['att_std']
    def_mu, def_std   = global_dists['def_mu'], global_dists['def_std']

    if team_ratings_df is not None:
        own_ratings = team_ratings_df[['team', 'id_fixture', 'final_attack_rating', 'final_defence_rating']].add_prefix('own_')
        opp_ratings = team_ratings_df[['team', 'id_fixture', 'final_attack_rating', 'final_defence_rating']].add_prefix('opp_')

        fixture_player_df = pd.merge(fixture_player_df, own_ratings, left_on=['team', 'id_fixture'], right_on=['own_team', 'own_id_fixture'], how='left')
        fixture_player_df = pd.merge(fixture_player_df, opp_ratings, left_on=['opponent', 'id_fixture'], right_on=['opp_team', 'opp_id_fixture'], how='left')

        # [FIX 6]: Utilize proper is_home masks for future fixtures
        if latest_team_ratings is not None:
            latest_map = latest_team_ratings.set_index('team')
            is_home_mask = fixture_player_df['is_home'].fillna(True)

            fixture_player_df['own_final_attack_rating'] = fixture_player_df['own_final_attack_rating'].fillna(
                pd.Series(np.where(is_home_mask, fixture_player_df['team'].map(latest_map['latest_final_attack_home']), fixture_player_df['team'].map(latest_map['latest_final_attack_away'])), index=fixture_player_df.index)
            )
            fixture_player_df['own_final_defence_rating'] = fixture_player_df['own_final_defence_rating'].fillna(
                pd.Series(np.where(is_home_mask, fixture_player_df['team'].map(latest_map['latest_final_defence_home']), fixture_player_df['team'].map(latest_map['latest_final_defence_away'])), index=fixture_player_df.index)
            )
            fixture_player_df['opp_final_attack_rating'] = fixture_player_df['opp_final_attack_rating'].fillna(
                pd.Series(np.where(~is_home_mask, fixture_player_df['opponent'].map(latest_map['latest_final_attack_home']), fixture_player_df['opponent'].map(latest_map['latest_final_attack_away'])), index=fixture_player_df.index)
            )
            fixture_player_df['opp_final_defence_rating'] = fixture_player_df['opp_final_defence_rating'].fillna(
                pd.Series(np.where(~is_home_mask, fixture_player_df['opponent'].map(latest_map['latest_final_defence_home']), fixture_player_df['opponent'].map(latest_map['latest_final_defence_away'])), index=fixture_player_df.index)
            )

        att_delta = fixture_player_df['own_final_attack_rating'] - fixture_player_df['opp_final_defence_rating']
        def_delta = fixture_player_df['opp_final_attack_rating'] - fixture_player_df['own_final_defence_rating']

        drop_cols = [c for c in fixture_player_df.columns if c.startswith('own_') or c.startswith('opp_')]
        fixture_player_df.drop(columns=drop_cols, inplace=True, errors='ignore')

    else:
        # FPL strength rating fallback
        strength_cols = ['id', 'strength_attack_home', 'strength_attack_away', 'strength_defence_home', 'strength_defence_away']
        opp_df = team_df[strength_cols].add_prefix('opp_')
        own_df = team_df[strength_cols].add_prefix('own_')
        fixture_player_df = pd.merge(fixture_player_df, opp_df, left_on='opponent', right_on='opp_id', how='left')
        fixture_player_df = pd.merge(fixture_player_df, own_df, left_on='team', right_on='own_id', how='left')

        # [FIX 1]: FPL API Cross-Mapping applying the strict inverse rule
        team_attack_rating  = np.where(fixture_player_df['is_home'], fixture_player_df['own_strength_attack_away'], fixture_player_df['own_strength_attack_home'])
        team_defence_rating = np.where(fixture_player_df['is_home'], fixture_player_df['own_strength_defence_away'], fixture_player_df['own_strength_defence_home'])

        opp_defence_rating  = np.where(~fixture_player_df['is_home'], fixture_player_df['opp_strength_defence_away'], fixture_player_df['opp_strength_defence_home'])
        opp_attack_rating   = np.where(~fixture_player_df['is_home'], fixture_player_df['opp_strength_attack_away'], fixture_player_df['opp_strength_attack_home'])

        att_delta = team_attack_rating - opp_defence_rating
        def_delta = opp_attack_rating  - team_defence_rating

        drop_cols = [c for c in fixture_player_df.columns if c.startswith('opp_strength') or c.startswith('own_strength') or c in ['opp_id', 'own_id']]
        fixture_player_df.drop(columns=drop_cols, inplace=True)

    # Shared multiplier calculation
    raw_att = 1 + (fixture_alpha_att * ((att_delta - att_mu) / att_std))
    raw_def = 1 + (fixture_alpha_def * ((def_delta - def_mu) / def_std))

    Z_CLIP = 3.0
    fixture_player_df['fixture_attack_multiplier'] = np.clip(raw_att, max(0.5, 1.0 - fixture_alpha_att * Z_CLIP), min(2.0, 1.0 + fixture_alpha_att * Z_CLIP))
    fixture_player_df['fixture_defence_multiplier'] = np.clip(raw_def, max(0.5, 1.0 - fixture_alpha_def * Z_CLIP), min(2.0, 1.0 + fixture_alpha_def * Z_CLIP))

    # 7. CALCULATE WALK-FORWARD MINUTES FEATURES
    fixture_player_df = compute_walkforward_minutes_features(
        fixture_player_df,
        raw_history_df,
        ema_alpha=params.get('minutes_ema_alpha', 0.40),
    )

    return fixture_player_df

# --- CELL 29 ---
def _fit_garch_minutes_volatility(
    raw_history_df: pd.DataFrame,
    min_history: int = 8,
) -> pd.DataFrame:
    """
    Fits GARCH(1,1) to minutes residuals to forecast conditional volatility.
    Falls back to sample standard deviation if 'arch' is not installed or fit fails.
    """
    try:
        from arch import arch_model
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning)
    except ImportError:
        # Fallback: Compute sample std per player
        res = raw_history_df.groupby('id_player')['minutes'].std().reset_index()
        res.columns = ['id_player', 'garch_cond_vol']
        res['garch_fitted'] = False
        return res

    results = []
    for player_id, group in raw_history_df.groupby('id_player'):
        mins = group['minutes'].values.astype(float)
        if len(mins) < min_history:
            results.append({'id_player': player_id, 'garch_cond_vol': np.std(mins) if len(mins) > 1 else 30.0, 'garch_fitted': False})
            continue
            
        try:
            # Fit GARCH(1,1) on minutes (using zero mean as rotation is a residual process)
            am = arch_model(mins, vol='Garch', p=1, q=1, dist='Normal', rescale=False)
            res = am.fit(update_freq=0, disp='off', show_warning=False)
            forecast = res.forecast(horizon=1)
            cond_vol = np.sqrt(forecast.variance.values[-1, 0])
            results.append({'id_player': player_id, 'garch_cond_vol': cond_vol, 'garch_fitted': True})
        except:
            results.append({'id_player': player_id, 'garch_cond_vol': np.std(mins), 'garch_fitted': False})
            
    return pd.DataFrame(results)
