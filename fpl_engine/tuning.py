import json
import os
import time
from datetime import datetime
import optuna
import pandas as pd
import numpy as np

# Suppress Optuna logging to avoid cluttering the dashboard
optuna.logging.set_verbosity(optuna.logging.WARNING)

async def auto_tune_if_needed(current_gw: int, force: bool = False, n_trials_override: int = None):
    """
    Checks if tuning is required. Triggers when:
      1. force=True (manual override)
      2. current_gw is 5+ weeks ahead of last_tuned_gw
      3. GW went backwards (new season detected, e.g. 37 → 1)
      4. last_tuned_date is more than 30 days ago
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(base_dir, 'tuned_params.json')
    
    if not os.path.exists(json_path):
        print("tuned_params.json not found, skipping auto-tune.")
        return
        
    with open(json_path, 'r') as f:
        params = json.load(f)
        
    last_tuned_gw = params.get('last_tuned_gw', 0)
    last_tuned_date_str = params.get('last_tuned_date', '2020-01-01T00:00:00')
    
    # Determine if tuning is needed
    gw_gap = current_gw - last_tuned_gw
    new_season = current_gw < last_tuned_gw  # GW went backwards → new season
    
    try:
        last_tuned_date = datetime.fromisoformat(last_tuned_date_str)
        days_since_tune = (datetime.now() - last_tuned_date).days
    except (ValueError, TypeError):
        days_since_tune = 999  # Force tune if date is unparseable
    
    stale = days_since_tune > 30
    needs_tune = force or gw_gap >= 5 or new_season or stale
    
    if needs_tune:
        reason = []
        if force: reason.append("forced")
        if gw_gap >= 5: reason.append(f"GW gap={gw_gap}")
        if new_season: reason.append(f"new season (GW {last_tuned_gw}→{current_gw})")
        if stale: reason.append(f"stale ({days_since_tune}d since last tune)")
        
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auto-Tune Triggered: {', '.join(reason)}")
        await _run_full_tuning_pipeline(params, json_path, current_gw, n_trials_override)
    else:
        print(f"Parameters up to date. (Last tuned GW: {last_tuned_gw}, {days_since_tune}d ago)")


def validate_and_fill_params(params: dict, contract: dict, silent: bool = False) -> dict:
    result = params.copy()
    filled = []
    for key, default in contract.items():
        if key not in result:
            result[key] = default
            filled.append(f"  {key} = {default}  <- filled from contract")

    if filled and not silent:
        print("WARNING: validate_and_fill_params missing keys auto-filled:")
        print("\n".join(filled))
    return result

def early_stopping_callback(study, trial, patience=40):
    if trial.number >= patience:
        best_trial_num = study.best_trial.number
        if trial.number - best_trial_num >= patience:
            print(f"\nStopping early: No improvement in the last {patience} trials.")
            study.stop()

def _get_stable_eval_df(df: pd.DataFrame, skip_gws: int = 4) -> pd.DataFrame:
    if 'season' in df.columns:
        min_gw_per_season = df.groupby('season')['gameweek'].transform('min')
        return df[df['gameweek'] >= min_gw_per_season + skip_gws]
    else:
        df = df.sort_values(['id_player', 'gameweek']).copy()
        gw_vals = df['gameweek'].values
        season_ids = np.cumsum(np.append(0, np.diff(gw_vals) < 0))
        df['__proxy_season'] = season_ids
        min_gw = df.groupby('__proxy_season')['gameweek'].transform('min')
        stable_df = df[df['gameweek'] >= min_gw + skip_gws].drop(columns=['__proxy_season'])
        return stable_df

def _reapply_minutes_ema(base_df: pd.DataFrame, params: dict, raw_history_df: pd.DataFrame) -> pd.DataFrame:
    alpha = params.get('minutes_ema_alpha', 0.40)
    df = base_df.copy()

    finished_fixture_ids = df[df['finished'] == True]['id_fixture'].unique()
    hist = raw_history_df.drop_duplicates(subset=['id_player', 'id_fixture'])
    hist = hist[hist['id_fixture'].isin(finished_fixture_ids)].sort_values(['id_player', 'kickoff_time'])

    hist['past_ema'] = (
        hist.groupby('id_player')['minutes']
        .transform(lambda x: x.shift(1).ewm(alpha=alpha, adjust=False).mean())
        .fillna(0)
    )

    hist['full_ema'] = (
        hist.groupby('id_player')['minutes']
        .ewm(alpha=alpha, adjust=False)
        .mean()
        .reset_index(level=0, drop=True)
    )
    latest_emas = hist.groupby('id_player')['full_ema'].last()

    past_map = hist.set_index(['id_player', 'id_fixture'])['past_ema']
    df['recent_minutes_form'] = df.set_index(['id_player', 'id_fixture']).index.map(past_map).values

    future_mask = df['recent_minutes_form'].isna()
    df.loc[future_mask, 'recent_minutes_form'] = df.loc[future_mask, 'id_player'].map(latest_emas).fillna(0)

    return df

def _get_gw_chunks(df: pd.DataFrame, n_chunks: int = 5) -> list[np.ndarray]:
    train_gws = sorted(df['gameweek'].unique())
    return [arr for arr in np.array_split(train_gws, n_chunks) if len(arr) > 0]


async def _run_full_tuning_pipeline(current_config: dict, json_path: str, current_gw: int, n_trials_override: int = None):
    from fpl_engine.config import load_tuned_params
    from fpl_engine.data import get_current_players_df, get_team_df, fetch_raw_history_cache, get_max_finished_gameweek
    from fpl_engine.features import (
        compute_rolling_team_ratings, blend_team_ratings, compute_global_z_distributions,
        get_fixture_players_stats_df
    )
    from fpl_engine.scoring import (
        _fit_regression_params, _fit_bonus_multinomial, _diagnose_bonus_model,
        _calculate_performance_indices
    )
    from fpl_engine.optimization import (
        get_averaged_production_params, minutes_composite_loss, calculate_overall_score
    )
    
    print("Starting FPL Engine Parameter Tuning (Extracted from Notebook)...")
    start_time = time.time()
    
    import warnings
    from optuna.exceptions import ExperimentalWarning
    warnings.filterwarnings("ignore", category=ExperimentalWarning)
    
    PARAM_CONTRACT = current_config['locked_params'].copy()
    PARAM_CONTRACT.update(current_config['adaptive_targets'])
    PARAM_CONTRACT.update(current_config['minutes_targets'])
    
    player_df         = get_current_players_df()
    active_player_ids = player_df['id'].unique()
    fpl_team_df       = get_team_df()

    raw_history_df = await fetch_raw_history_cache(active_player_ids, use_cache=True)
    bonus_model  = _fit_bonus_multinomial(raw_history_df)
    _diagnose_bonus_model(bonus_model)

    BASE_CONSTANTS = PARAM_CONTRACT.copy()

    _base_roll, _base_latest = compute_rolling_team_ratings(
        raw_history_df, ema_alpha=BASE_CONSTANTS.get('rolling_ema_alpha', 0.26)
    )
    base_team_ratings, base_latest_ratings = blend_team_ratings(
        _base_roll, _base_latest, fpl_team_df,
        league_avg_xG=BASE_CONSTANTS.get('league_avg_xG', 1.45),
        league_avg_xGC=BASE_CONSTANTS.get('league_avg_xGC', 1.45),
        blend_alpha=BASE_CONSTANTS.get('blend_alpha', 0.5),
        min_fixtures_full_trust=BASE_CONSTANTS.get('min_fixtures_full_trust', 15)
    )

    global_dists = compute_global_z_distributions(base_team_ratings)

    MAX_GW       = get_max_finished_gameweek()
    if MAX_GW is None: MAX_GW = 38
    TRAIN_CUTOFF = MAX_GW - 5

    static_base_df = get_fixture_players_stats_df(
        BASE_CONSTANTS, raw_history_df, global_dists,
        team_ratings_df=base_team_ratings,
        latest_team_ratings=base_latest_ratings
    )
    train_base_df  = static_base_df[static_base_df['gameweek'] <= TRAIN_CUTOFF]

    base_reg_params = _fit_regression_params(train_base_df)
    BASE_CONSTANTS.update(base_reg_params)

    def run_minutes_optimization(base_params, precomputed_df, raw_history_df, n_trials=100, seed=42):
        def objective_minutes(trial):
            params = base_params.copy()
            params.update({
                'minutes_w_form':           trial.suggest_float('minutes_w_form',           0.50, 1.00),
                'minutes_w_haaland_season': trial.suggest_float('minutes_w_haaland_season', 0.10, 0.98),
                'minutes_role_floor':       trial.suggest_float('minutes_role_floor',       0.05, 0.50),
                'minutes_ema_alpha':        trial.suggest_float('minutes_ema_alpha',        0.20, 0.95),
                'minutes_loyalty_w':        trial.suggest_float('minutes_loyalty_w',        0.40, 0.95),
                'minutes_trend_scale':      trial.suggest_float('minutes_trend_scale',      0.10, 1.20),
                'minutes_high_streak':      trial.suggest_float('minutes_high_streak',      2.00, 6.00),
                'minutes_low_vol_thresh':   trial.suggest_float('minutes_low_vol_thresh',   5.00, 45.0),
            })
            params = validate_and_fill_params(params, PARAM_CONTRACT, silent=True)

            trial_df = _reapply_minutes_ema(precomputed_df, params, raw_history_df)
            temp_df  = _calculate_performance_indices(
                trial_df[trial_df['gameweek'] <= TRAIN_CUTOFF], params, bonus_model=bonus_model
            )

            eval_df = temp_df[
                (temp_df['chance_of_playing_next_round'].fillna(100) == 100) &
                (temp_df['finished'] == True)
            ]

            eval_df = _get_stable_eval_df(eval_df, skip_gws=4)
            gw_chunks = _get_gw_chunks(eval_df, n_chunks=5)

            cumulative_gws = []
            last_loss = None

            for step, chunk in enumerate(gw_chunks):
                cumulative_gws.extend(chunk.tolist())
                chunk_eval = eval_df[eval_df['gameweek'].isin(cumulative_gws)]

                if len(chunk_eval) == 0:
                    continue

                y_true_mins = chunk_eval['actual_minutes'].values
                y_pred_mins = chunk_eval['minutes_IDX'].values
                
                # Cap rest game penalties instead of downweighting the whole sample
                rest_mask = (y_true_mins == 0) & (chunk_eval['start_per_gameplayed'] > 0.80).values & (chunk_eval['minutes_per_game'] > 75).values
                y_pred_capped = np.where(rest_mask, np.minimum(y_pred_mins, 30), y_pred_mins)

                loss = minutes_composite_loss(
                    y_true_mins,
                    y_pred_capped,
                    sample_weight=np.ones_like(y_true_mins, dtype=float)
                )
                trial.report(loss, step)
                last_loss = loss

                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return last_loss

        pruner_p1 = optuna.pruners.PatientPruner(
            optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1, interval_steps=1),
            patience=1
        )
        study = optuna.create_study(direction='minimize', pruner=pruner_p1)
        study.optimize(objective_minutes, n_trials=n_trials, n_jobs=1, show_progress_bar=True, callbacks=[lambda study, trial: early_stopping_callback(study, trial, patience=15, direction='minimize')])
        best = get_averaged_production_params(study, top_k=5, primary_metric_idx=0, maximize_primary=False)
        return {k: round(round(v / 0.005) * 0.005, 3) for k, v in best.items()} if best else {}

    print("Phase 1: Minutes Optimization...")
    best_minutes_v1 = run_minutes_optimization(BASE_CONSTANTS, static_base_df, raw_history_df, n_trials=n_trials_override or 40)
    
    def objective_alphas(trial):
        params = {**BASE_CONSTANTS, **best_minutes_v1}
        params.update({
            'recency_ema_alpha' : trial.suggest_float('recency_ema_alpha', 0.01, 0.40),
            'rolling_ema_alpha' : trial.suggest_float('rolling_ema_alpha', 0.05, 0.40),
        })
        params = validate_and_fill_params(params, PARAM_CONTRACT, silent=True)

        _trial_rolling, _trial_latest = compute_rolling_team_ratings(raw_history_df, ema_alpha=params['rolling_ema_alpha'])
        _team_ratings, _latest = blend_team_ratings(
            _trial_rolling, _trial_latest, fpl_team_df,
            league_avg_xG=params['league_avg_xG'], league_avg_xGC=params['league_avg_xGC'],
            blend_alpha=params['blend_alpha'], min_fixtures_full_trust=params['min_fixtures_full_trust'],
        )
        _global_dists = compute_global_z_distributions(_team_ratings)

        trial_df = get_fixture_players_stats_df(
            params, raw_history_df, _global_dists,
            team_ratings_df=_team_ratings, latest_team_ratings=_latest,
        )
        train_df = trial_df[trial_df['gameweek'] <= TRAIN_CUTOFF]

        dynamic_reg = _fit_regression_params(train_df)
        params.update(dynamic_reg)

        temp_df   = _calculate_performance_indices(train_df, params, bonus_model=bonus_model)
        eval_base = temp_df[temp_df['actual_minutes'] > 0]
        eval_base = _get_stable_eval_df(eval_base, skip_gws=4)

        gw_chunks      = _get_gw_chunks(eval_base, n_chunks=3)
        cumulative_gws = []
        last_score     = None

        for step, chunk in enumerate(gw_chunks):
            cumulative_gws.extend(chunk.tolist())
            chunk_eval = eval_base[eval_base['gameweek'].isin(cumulative_gws)]
            if len(chunk_eval) == 0:
                continue
            score = calculate_overall_score(chunk_eval, 'actual_points', 'fixture_calibrated_points')
            trial.report(score, step)
            last_score = score
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return last_score

    print("\nPhase 2: Alpha + Recency Optimization...")
    pruner_p2 = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0, interval_steps=1)
    study_alphas = optuna.create_study(direction='maximize', pruner=pruner_p2)
    study_alphas.optimize(objective_alphas, n_trials=n_trials_override or 30, n_jobs=1, show_progress_bar=False, callbacks=[lambda study, trial: early_stopping_callback(study, trial, patience=15)])
    
    best_alphas = get_averaged_production_params(study_alphas, top_k=3, primary_metric_idx=0, maximize_primary=True)
    if best_alphas:
        best_alphas = {k: round(round(v / 0.005) * 0.005, 3) for k, v in best_alphas.items()}
    else:
        best_alphas = {}

    print("\nPhase 3: Perf Index Tuning...")
    master_params = BASE_CONSTANTS.copy()
    master_params.update(best_minutes_v1)
    master_params.update(best_alphas)
    master_params = validate_and_fill_params(master_params, PARAM_CONTRACT)

    _master_roll, _master_latest = compute_rolling_team_ratings(raw_history_df, ema_alpha=master_params.get('rolling_ema_alpha', 0.26))
    master_team_ratings, master_latest = blend_team_ratings(
        _master_roll, _master_latest, fpl_team_df,
        league_avg_xG=master_params['league_avg_xG'], league_avg_xGC=master_params['league_avg_xGC'],
        blend_alpha=master_params['blend_alpha'], min_fixtures_full_trust=master_params['min_fixtures_full_trust']
    )
    master_dists = compute_global_z_distributions(master_team_ratings)

    static_perf_df  = get_fixture_players_stats_df(
        master_params, raw_history_df, master_dists,
        team_ratings_df=master_team_ratings, latest_team_ratings=master_latest
    )
    train_perf_df   = static_perf_df[static_perf_df['gameweek'] <= TRAIN_CUTOFF]
    pre_fit_reg_params = _fit_regression_params(train_perf_df)
    master_params.update(pre_fit_reg_params)

    def objective_perf_idx(trial):
        params = master_params.copy()
        params.update({
            'c_finish':       trial.suggest_float('c_finish',       0.5,  15.0),
            'c_protect':      trial.suggest_float('c_protect',      0.5,  15.0),
            'c_base_defense': trial.suggest_float('c_base_defense', 0.5,  50.0),
        })
        params = validate_and_fill_params(params, PARAM_CONTRACT, silent=True)

        temp_df   = _calculate_performance_indices(train_perf_df, params, bonus_model=bonus_model)
        eval_base = temp_df[temp_df['actual_minutes'] > 0]
        eval_base = _get_stable_eval_df(eval_base, skip_gws=4)

        gw_chunks = _get_gw_chunks(eval_base, n_chunks=5)
        cumulative_gws = []
        last_score = None

        for step, chunk in enumerate(gw_chunks):
            cumulative_gws.extend(chunk.tolist())
            chunk_eval = eval_base[eval_base['gameweek'].isin(cumulative_gws)]
            if len(chunk_eval) == 0:
                continue
            score = calculate_overall_score(chunk_eval, 'actual_points', 'fixture_calibrated_points')
            trial.report(score, step)
            last_score = score
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        return last_score

    pruner_p3 = optuna.pruners.PatientPruner(optuna.pruners.PercentilePruner(percentile=75.0, n_startup_trials=10, n_warmup_steps=1, interval_steps=1), patience=1)
    study_perf = optuna.create_study(direction='maximize', pruner=pruner_p3)
    study_perf.optimize(objective_perf_idx, n_trials=n_trials_override or 40, n_jobs=1, show_progress_bar=False, callbacks=[lambda study, trial: early_stopping_callback(study, trial, patience=15)])

    final_perf_params = get_averaged_production_params(study_perf, top_k=5, primary_metric_idx=0, maximize_primary=True)
    if final_perf_params:
        final_perf_params = {k: round(round(v / 0.005) * 0.005, 3) for k, v in final_perf_params.items()}
    else:
        final_perf_params = {}

    final_param = PARAM_CONTRACT.copy()
    final_param.update(BASE_CONSTANTS)
    final_param.update(best_minutes_v1)
    final_param.update(best_alphas)
    final_param.update(final_perf_params)

    # Save logic
    for k in current_config['minutes_targets'].keys():
        if k in final_param:
            current_config['minutes_targets'][k] = final_param[k]

    for k in current_config['adaptive_targets'].keys():
        if k in final_param:
            current_config['adaptive_targets'][k] = final_param[k]
            
    for k in current_config['locked_params'].keys():
        if k in final_param:
            current_config['locked_params'][k] = final_param[k]

    current_config['last_tuned_gw'] = current_gw
    current_config['last_tuned_date'] = datetime.now().isoformat()
    
    with open(json_path, 'w') as f:
        json.dump(current_config, f, indent=4)
        
    print(f"Tuning complete in {time.time() - start_time:.2f}s. Parameters updated in tuned_params.json")
