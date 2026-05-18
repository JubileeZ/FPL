import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import nest_asyncio
import contextlib
import io
import os
import sys
import pulp

# Ensure event loop handles nesting in Streamlit environments
nest_asyncio.apply()

# Core fpl_engine imports
from fpl_engine.config import MY_FPL_ID, get_season_params, load_tuned_params
from fpl_engine.data import (
    get_current_players_df, fetch_raw_history_cache,
    get_team_df, get_pos_constraint_df,
    get_current_gameweek, get_max_finished_gameweek,
    get_fpl_gameweek_data, get_my_player_ids,
    get_dynamic_weights,
)
from fpl_engine.features import (
    compute_rolling_team_ratings, blend_team_ratings,
    get_fixture_players_stats_df, compute_global_z_distributions,
)
from fpl_engine.scoring import (
    _fit_bonus_multinomial, _fit_regression_params,
    _calculate_performance_indices, create_optimized_custom_score,
)
from fpl_engine.solver import plan_sequential_transfers
from fpl_engine.tuning import auto_tune_if_needed

# =========================================================================
# PAGE SETUP & PREMIUM STYLING
# =========================================================================
st.set_page_config(
    page_title="FPL Squad Optimizer",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium dark glassmorphic theme and styling
st.markdown("""
<style>
    /* Global style modifications */
    .stApp {
        background: linear-gradient(135deg, #0e0b1f 0%, #151130 100%);
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Premium Header */
    .header-container {
        text-align: center;
        padding: 1.5rem 1rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #ff007f 0%, #00f2fe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        color: #a0aec0;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Cards and Glassmorphism */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 16px 0 rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(8px);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #00f2fe;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Custom buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #ff007f 0%, #7928ca 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(255, 0, 127, 0.3) !important;
        transition: all 0.3s ease !important;
        width: 100%;
    }
    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 0, 127, 0.5) !important;
    }
    
    /* Styled lists and lists inside tables */
    .transfer-arrow {
        color: #ff007f;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Custom tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: rgba(255, 255, 255, 0.02);
        padding: 6px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.08) !important;
        color: #00f2fe !important;
    }
</style>
""", unsafe_allow_html=True)

# Title Header
st.markdown("""
<div class="header-container">
    <div class="header-title">FPL Squad Optimizer</div>
    <div class="header-subtitle">Advanced Mathematical Transfer & Lineup Optimization</div>
</div>
""", unsafe_allow_html=True)


# =========================================================================
# ASYNC DATA FETCHING & PIPELINE ENGINE
# =========================================================================
async def run_fpl_pipeline(manager_id, force_retune=False):
    """Executes the full FPL projection calculation pipeline asynchronously."""
    status_box = st.empty()
    
    def update_status(text, pct):
        status_box.markdown(f"""
        <div class="glass-card" style="text-align: center;">
            <div style="font-weight:600; margin-bottom: 0.5rem; color:#00f2fe;">{text}</div>
            <div style="background-color: rgba(255,255,255,0.05); border-radius: 10px; height: 10px;">
                <div style="background: linear-gradient(90deg, #ff007f, #00f2fe); width: {pct}%; height: 10px; border-radius: 10px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Get gameweek numbers
    update_status("Initializing gameweek details...", 10)
    current_gw = get_current_gameweek()
    data_gameweek = get_max_finished_gameweek()
    
    if current_gw is None:
        current_gw = 38
    if data_gameweek is None:
        data_gameweek = 37

    # 2. Automated Parameter Tuning
    update_status("Evaluating parameter staleness & executing Optuna tuning...", 25)
    await auto_tune_if_needed(current_gw, force=force_retune)
    params = get_season_params(current_gw)

    # 3. Base FPL structures
    update_status("Fetching player profiles and team rosters...", 40)
    fpl_team_df = get_team_df()
    players_df = get_current_players_df()

    # 4. Asynchronous Match History Fetching
    update_status("Loading raw match history from API / cache...", 55)
    active_player_ids = players_df[players_df['minutes'] > 0]['id'].tolist()
    
    # Check if assets directory is accessible (handles Sandbox restrictions)
    use_cache = True
    if not os.path.exists("assets"):
        try:
            os.makedirs("assets", exist_ok=True)
        except Exception:
            use_cache = False  # Bypasses parquet caching if permissions block it
            
    raw_history_df = await fetch_raw_history_cache(active_player_ids, use_cache=use_cache)

    # 5. Team Ratings & Strengths
    update_status("Computing walk-forward EMA team ratings...", 70)
    rolling_ratings_raw, latest_ratings_raw = compute_rolling_team_ratings(
        raw_history_df,
        ema_alpha=params.get('rolling_ema_alpha', 0.15),
        min_fixtures=3,
    )
    team_ratings_df, latest_team_ratings = blend_team_ratings(
        rolling_ratings_raw,
        latest_ratings_raw,
        fpl_team_df,
        league_avg_xG=params.get('league_avg_xG', 1.45),
        league_avg_xGC=params.get('league_avg_xGC', 1.45),
        blend_alpha=params.get('blend_alpha', 0.75),
        min_fixtures_full_trust=params.get('min_fixtures_full_trust', 10),
    )

    # 6. Fixture Player Projections
    update_status("Simulating fixture match-ups and duel indexes...", 80)
    global_dists = compute_global_z_distributions(team_ratings_df)
    fixture_player_df = get_fixture_players_stats_df(
        params,
        raw_history_df,
        global_dists,
        team_ratings_df=team_ratings_df,
        latest_team_ratings=latest_team_ratings,
    )

    # 7. Model Fitting & Performance Calculations
    update_status("Fitting multinomial bonus models & projecting raw scores...", 90)
    reg_params = _fit_regression_params(fixture_player_df)
    bonus_model = _fit_bonus_multinomial(raw_history_df)
    params.update(reg_params)

    fixture_player_df = _calculate_performance_indices(
        fixture_player_df,
        params,
        bonus_model=bonus_model
    )

    # 8. Aggregating to per-GW Projections
    update_status("Finalizing player custom scores and dynamic weight mappings...", 95)
    grouping_columns = [
        'gameweek', 'id_player', 'now_cost', 'selected_by_percent',
        'web_name', 'position', 'team_name'
    ]
    sum_columns = [
        'Perf_IDX', 'ceiling_score', 'GOAL_INDEX', 'ASSIST_INDEX',
        'CLEAN_SHEET_INDEX', 'bonus_component', 'defcon_component',
        'minutes_IDX', 'actual_minutes'
    ]
    mean_columns = [
        'recent_minutes_form', 'finishing_factor', 'protective_factor',
        'fixture_attack_multiplier', 'fixture_defence_multiplier',
        'fixture_calibrated_points',
        'start_per_gameplayed', 'consecutive_start_streak', 'hybrid_bps_abs', 'score_std',
    ]

    agg_dict = {col: 'sum' for col in sum_columns if col in fixture_player_df.columns}
    agg_dict.update({col: 'mean' for col in mean_columns if col in fixture_player_df.columns})
    valid_grouping = [c for c in grouping_columns if c in fixture_player_df.columns]

    gw_projection_df = fixture_player_df.groupby(valid_grouping).agg(agg_dict).reset_index()

    # Get Manager's Gameweek data & dynamic weights
    try:
        fpl_gameweek_data = get_fpl_gameweek_data(manager_id)
        weights = get_dynamic_weights(
            fpl_gameweek_data, 
            data_gameweek,
            max_diff_weight=0.13,
            max_upside_weight=0.12
        )
    except Exception:
        weights = {
            'diff_weight': 0.13,
            'upside_weight': 0.12,
            'mode': 'DEFAULT (Offline/Error)',
            'rank_pct': 50.0
        }

    status_box.empty()
    
    return {
        'gw_projection_df': gw_projection_df,
        'current_gw': current_gw,
        'data_gameweek': data_gameweek,
        'weights': weights,
        'params': params,
        'players_df': players_df
    }


# =========================================================================
# STATE INITIALIZATION
# =========================================================================
if 'pipeline_data' not in st.session_state:
    st.session_state['pipeline_data'] = None

if 'manager_id' not in st.session_state:
    st.session_state['manager_id'] = MY_FPL_ID

# Trigger data loading if empty
if st.session_state['pipeline_data'] is None:
    with st.spinner("Fetching official FPL APIs and bootstrapping models..."):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            st.session_state['pipeline_data'] = loop.run_until_complete(
                run_fpl_pipeline(st.session_state['manager_id'])
            )
        except Exception as e:
            st.error(f"Failed to fetch FPL API data: {e}. Running with cached/mock profile.")
            # Fallback to local default parameters to prevent crashes
            st.session_state['pipeline_data'] = {
                'gw_projection_df': pd.DataFrame(),
                'current_gw': 30,
                'data_gameweek': 29,
                'weights': {'diff_weight': 0.13, 'upside_weight': 0.12, 'mode': 'OFFLINE FALLBACK', 'rank_pct': 50.0},
                'params': {},
                'players_df': pd.DataFrame()
            }

pipe_data = st.session_state['pipeline_data']
current_gw = pipe_data['current_gw']
players_df = pipe_data['players_df']
gw_projection_df = pipe_data['gw_projection_df']
default_weights = pipe_data['weights']

# Pre-populate active player dictionaries for Locked/Banned Multi-selects
player_select_options = []
player_id_map = {}
if not players_df.empty:
    sorted_players = players_df.sort_values(by='web_name')
    for _, row in sorted_players.iterrows():
        label = f"{row['web_name']} ({row['position']} - {row['team_name']})"
        player_select_options.append(label)
        player_id_map[label] = int(row['id'])

# =========================================================================
# SIDEBAR CONTROLS & OVERRIDES
# =========================================================================
st.sidebar.markdown('<div style="font-size:1.2rem; font-weight:700; color:#00f2fe; margin-bottom:1rem;">Optimization Parameters</div>', unsafe_allow_html=True)

# 1. Profile Manager ID
manager_id = st.sidebar.number_input(
    "FPL Manager ID",
    min_value=1,
    value=st.session_state['manager_id'],
    help="Enter your official Fantasy Premier League manager ID."
)

if manager_id != st.session_state['manager_id']:
    st.session_state['manager_id'] = manager_id
    if st.sidebar.button("Reload Manager Profile"):
        st.session_state['pipeline_data'] = None
        st.rerun()

# 2. Budget & Transfer Settings
st.sidebar.markdown("---")
planning_horizon = st.sidebar.slider("Planning Horizon (Weeks)", min_value=1, max_value=8, value=6)
bank_balance = st.sidebar.number_input("Available Bank (£M)", min_value=0.0, max_value=25.0, value=3.0, step=0.1)
free_transfers = st.sidebar.slider("Available Free Transfers", min_value=1, max_value=5, value=1)

# 3. Dynamic Weight Adjustments
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Calculated Weight Mode:** `{default_weights.get('mode', 'N/A')}`")
override_weights = st.sidebar.checkbox("Override Solver Weights", value=False)
if override_weights:
    diff_weight = st.sidebar.slider("Differential Penalty Weight", min_value=0.0, max_value=0.30, value=default_weights.get('diff_weight', 0.13), step=0.01)
    upside_weight = st.sidebar.slider("Captain Upside Weight", min_value=0.0, max_value=0.30, value=default_weights.get('upside_weight', 0.12), step=0.01)
else:
    diff_weight = default_weights.get('diff_weight', 0.13)
    upside_weight = default_weights.get('upside_weight', 0.12)

# 4. Locking & Banning Players
st.sidebar.markdown("---")
fixed_players_selected = st.sidebar.multiselect("Lock Players in Squad", options=player_select_options)
banned_players_selected = st.sidebar.multiselect("Ban Players from Solver", options=player_select_options)

fixed_player_ids = [player_id_map[p] for p in fixed_players_selected if p in player_id_map]
banned_player_ids = [player_id_map[p] for p in banned_players_selected if p in player_id_map]

st.sidebar.markdown("---")
force_reoptimize = st.sidebar.button("⚡ Force Optuna Re-tune")
if force_reoptimize:
    with st.spinner("Re-executing hyperparameter optimization study..."):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        st.session_state['pipeline_data'] = loop.run_until_complete(
            run_fpl_pipeline(manager_id, force_retune=True)
        )
        st.rerun()

# =========================================================================
# APP SUMMARY CARDS
# =========================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-label">Current Gameweek</div>
        <div class="metric-value">GW {current_gw}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-label">Manager ID</div>
        <div class="metric-value">{manager_id}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-label">Manager Percentile</div>
        <div class="metric-value">{default_weights.get('rank_pct', 50.0):.1f}%</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
    st.markdown(f"""
    <div class="glass-card" style="text-align: center;">
        <div class="metric-label">Optuna Status</div>
        <div class="metric-value" style="color: #ff007f;">ACTIVE</div>
    </div>
    """, unsafe_allow_html=True)


# Apply custom score calculation to aggregated dataframe
if not gw_projection_df.empty:
    gw_projection_df = create_optimized_custom_score(
        df=gw_projection_df,
        differential_weight=diff_weight,
        upside_weight=upside_weight,
        visualize=False
    )

# Create Main Tabs
tab_transfers, tab_leaders, tab_my_team, tab_diagnostics = st.tabs([
    "🔄 Transfer Planner", 
    "🌟 Position Leaders", 
    "📋 My Current Team", 
    "🛠️ Optimization Logs"
])

# =========================================================================
# TAB 1: SEQUENTIAL TRANSFER PLANNER
# =========================================================================
with tab_transfers:
    st.markdown("<h3 style='color:#00f2fe; margin-bottom: 1.5rem;'>Sequential Solver Recommendations</h3>", unsafe_allow_html=True)
    
    # Run the solver on user button click or immediately if pipeline exists
    if gw_projection_df.empty:
        st.warning("Please load or verify the FPL data connection to run the planner.")
    else:
        # Fetch current manager picks
        try:
            my_current_team_ids = get_my_player_ids(manager_id, current_gw)
        except Exception:
            my_current_team_ids = []
            
        if not my_current_team_ids:
            st.error("Could not fetch player picks for this Manager ID. Make sure the ID is correct and active.")
        else:
            # Gather player realizable values
            # Streamlit sidebar inputs are used to create the realizable value dict
            current_realizable_value_dict = {}
            if not players_df.empty:
                my_squad_df = players_df[players_df['id'].isin(my_current_team_ids)]
                for _, row in my_squad_df.iterrows():
                    current_realizable_value_dict[int(row['id'])] = float(row['now_cost'])
            
            # Setup inputs for Pulp solver
            solver_fixed = {'Default': []}
            solver_banned = {'Default': [183, 221, 367, 126] + banned_player_ids}
            if fixed_player_ids:
                solver_fixed['Default'] = fixed_player_ids
            
            # Capture Pulp prints to stdout to render in streamlit
            stdout_buffer = io.StringIO()
            with contextlib.redirect_stdout(stdout_buffer):
                solver_results = plan_sequential_transfers(
                    gw_projection_df=gw_projection_df,
                    current_team_ids=my_current_team_ids,
                    start_gameweek=current_gw + 1,
                    planning_horizon=planning_horizon,
                    initial_free_transfers=free_transfers,
                    current_realizable_value_dict=current_realizable_value_dict,
                    bank_balance=bank_balance,
                    ft_value=1.23,
                    bench_factor=1e-4,
                    objective_column='custom_score',
                    captain_column='ceiling_score',
                    fixed_player_dict=solver_fixed,
                    banned_player_dict=solver_banned,
                    return_model=True
                )
            
            captured_stdout = stdout_buffer.getvalue()
            
            # Display detailed transfer schedules
            if solver_results:
                prob, variables = solver_results
                squad = variables['squad']
                starters = variables['starters']
                captain = variables['captain']
                trans_in = variables['trans_in']
                player_ids = variables['player_ids']
                
                gameweeks = list(range(current_gw + 1, np.minimum(current_gw + 1 + planning_horizon, 39)))
                player_details_df = gw_projection_df.drop_duplicates('id_player').set_index('id_player')
                
                st.success(f"Optimal sequential transfer schedule successfully planned! Utility Index: {pulp.value(prob.objective):.2f}")
                
                for t in gameweeks:
                    with st.expander(f"📅 Gameweek {t} Transfer & Lineup Strategy", expanded=(t == current_gw + 1)):
                        # Identify transfers in/out
                        trans_in_ids = [p for p in player_ids if (trans_in[p][t].varValue or 0) > 0.9]
                        squad_ids = [p for p in player_ids if squad[p][t].varValue > 0.9]
                        starter_ids = [p for p in player_ids if starters[p][t].varValue > 0.9]
                        bench_ids = list(set(squad_ids) - set(starter_ids))
                        
                        captain_id = max(player_ids, key=lambda p: captain[p][t].varValue or 0)
                        
                        # Show transfers visually
                        if trans_in_ids:
                            st.markdown("##### 🔄 Transfers Suggested")
                            out_cols = st.columns(len(trans_in_ids))
                            for idx, in_id in enumerate(trans_in_ids):
                                in_player = player_details_df.loc[in_id]
                                # Simple heuristic to match cost out
                                out_id = squad_ids[idx] if idx < len(squad_ids) else squad_ids[0]
                                out_player = player_details_df.loc[out_id]
                                
                                with out_cols[idx]:
                                    st.markdown(f"""
                                    <div class="glass-card" style="border-left: 4px solid #ff007f;">
                                        <div style="font-size:0.75rem; color:#a0aec0; text-transform:uppercase;">Out</div>
                                        <div style="font-weight:700; color:#e2e8f0;">{out_player['web_name']} ({out_player['position']})</div>
                                        <div style="font-size:0.8rem; color:#ff007f;">£{out_player['now_cost']:.1f}M</div>
                                        <div style="text-align:center; margin: 0.3rem 0;" class="transfer-arrow">⬇️</div>
                                        <div style="font-size:0.75rem; color:#a0aec0; text-transform:uppercase;">In</div>
                                        <div style="font-weight:700; color:#00f2fe;">{in_player['web_name']} ({in_player['position']})</div>
                                        <div style="font-size:0.8rem; color:#00f2fe;">£{in_player['now_cost']:.1f}M</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("No transfers suggested for this Gameweek. Save the free transfer! 🛑")
                        
                        # Show XI and Bench
                        st.markdown("##### ⚽ Projected Lineup")
                        col_starter, col_bench = st.columns([2, 1])
                        
                        with col_starter:
                            st.markdown("**Starting XI**")
                            xi_df = player_details_df.loc[starter_ids].copy()
                            xi_df['is_captain'] = (xi_df.index == captain_id).astype(int)
                            xi_df['role'] = np.where(xi_df['is_captain'] == 1, "Captain (C)", "Starter")
                            xi_df['display_name'] = np.where(xi_df['is_captain'] == 1, xi_df['web_name'] + " 👑", xi_df['web_name'])
                            
                            position_order = ['GKP', 'DEF', 'MID', 'FWD']
                            xi_df['position'] = pd.Categorical(xi_df['position'], categories=position_order, ordered=True)
                            xi_df = xi_df.sort_values(by=['position', 'custom_score'], ascending=[True, False])
                            
                            st.dataframe(
                                xi_df[['display_name', 'position', 'team_name', 'now_cost', 'custom_score', 'role']],
                                column_config={
                                    "display_name": "Player Name",
                                    "position": "Pos",
                                    "team_name": "Team",
                                    "now_cost": "Cost (£M)",
                                    "custom_score": "Proj Points",
                                    "role": "Role"
                                },
                                hide_index=True,
                                use_container_width=True
                            )
                            
                        with col_bench:
                            st.markdown("**Bench**")
                            bench_df = player_details_df.loc[bench_ids].copy()
                            bench_df['position'] = pd.Categorical(bench_df['position'], categories=position_order, ordered=True)
                            bench_df = bench_df.sort_values(by=['position', 'custom_score'], ascending=[True, False])
                            
                            st.dataframe(
                                bench_df[['web_name', 'position', 'team_name', 'now_cost', 'custom_score']],
                                column_config={
                                    "web_name": "Player Name",
                                    "position": "Pos",
                                    "team_name": "Team",
                                    "now_cost": "Cost",
                                    "custom_score": "Points"
                                },
                                hide_index=True,
                                use_container_width=True
                            )
            else:
                st.error("Optimization pipeline failed. Check locked and banned player parameter conflicts.")


# =========================================================================
# TAB 2: POSITION LEADERS
# =========================================================================
with tab_leaders:
    st.markdown("<h3 style='color:#00f2fe;'>Projected Position Leaders</h3>", unsafe_allow_html=True)
    st.markdown("Filter and view the top performing players across each position calibrated on their multi-gameweek expectations.")
    
    if gw_projection_df.empty:
        st.warning("Please load or verify the FPL data connection to view player projections.")
    else:
        # Filter projections for current and upcoming gameweeks
        lead_df = gw_projection_df[gw_projection_df['gameweek'] >= current_gw + 1].copy()
        
        projection_group = ['id_player', 'now_cost', 'selected_by_percent', 'web_name', 'position', 'team_name']
        avg_cols = ['custom_score', 'ceiling_score', 'minutes_IDX', 'Perf_IDX', 'GOAL_INDEX', 'ASSIST_INDEX', 'CLEAN_SHEET_INDEX']
        avg_cols = [c for c in avg_cols if c in lead_df.columns]
        
        # Aggregate across the planning horizon
        leader_totals = lead_df.groupby(projection_group)[avg_cols].mean().reset_index()
        
        # Sub-tabs for positions
        sub_gkp, sub_def, sub_mid, sub_fwd = st.tabs(["🧤 GKP", "🛡️ DEF", "🏃 MID", "⚔️ FWD"])
        
        position_list = ['GKP', 'DEF', 'MID', 'FWD']
        sub_tabs = [sub_gkp, sub_def, sub_mid, sub_fwd]
        
        for pos, tab in zip(position_list, sub_tabs):
            with tab:
                pos_leaders = leader_totals[leader_totals['position'] == pos].copy()
                pos_leaders = pos_leaders.sort_values(by='custom_score', ascending=False).head(15)
                
                if pos_leaders.empty:
                    st.info(f"No projections found for position {pos}")
                else:
                    st.dataframe(
                        pos_leaders[['web_name', 'team_name', 'now_cost', 'selected_by_percent', 'minutes_IDX', 'custom_score']],
                        column_config={
                            "web_name": "Player Name",
                            "team_name": "Team",
                            "now_cost": "Cost (£M)",
                            "selected_by_percent": "Own (%)",
                            "minutes_IDX": "Expected Minutes",
                            "custom_score": "Calibrated Score"
                        },
                        hide_index=True,
                        use_container_width=True
                    )


# =========================================================================
# TAB 3: MY CURRENT SQUAD
# =========================================================================
with tab_my_team:
    st.markdown("<h3 style='color:#00f2fe;'>My Current Squad</h3>", unsafe_allow_html=True)
    
    if players_df.empty:
        st.warning("Please load or verify the FPL data connection to view your current squad.")
    else:
        try:
            my_current_team_ids = get_my_player_ids(manager_id, current_gw)
        except Exception:
            my_current_team_ids = []
            
        if not my_current_team_ids:
            st.info("Please verify that your FPL Manager ID is correct to dynamically fetch your lineup.")
        else:
            my_squad_full = players_df[players_df['id'].isin(my_current_team_ids)].copy()
            position_order = ['GKP', 'DEF', 'MID', 'FWD']
            my_squad_full['position'] = pd.Categorical(my_squad_full['position'], categories=position_order, ordered=True)
            my_squad_full = my_squad_full.sort_values(by=['position', 'total_points'], ascending=[True, False])
            
            # Simple UI separation for squad view
            gkps = my_squad_full[my_squad_full['position'] == 'GKP']
            defs = my_squad_full[my_squad_full['position'] == 'DEF']
            mids = my_squad_full[my_squad_full['position'] == 'MID']
            fwds = my_squad_full[my_squad_full['position'] == 'FWD']
            
            st.markdown("#### 🧤 Goalkeepers")
            st.dataframe(gkps[['web_name', 'team_name', 'now_cost', 'selected_by_percent', 'total_points']], hide_index=True, use_container_width=True)
            
            st.markdown("#### 🛡️ Defenders")
            st.dataframe(defs[['web_name', 'team_name', 'now_cost', 'selected_by_percent', 'total_points']], hide_index=True, use_container_width=True)
            
            st.markdown("#### 🏃 Midfielders")
            st.dataframe(mids[['web_name', 'team_name', 'now_cost', 'selected_by_percent', 'total_points']], hide_index=True, use_container_width=True)
            
            st.markdown("#### ⚔️ Forwards")
            st.dataframe(fwds[['web_name', 'team_name', 'now_cost', 'selected_by_percent', 'total_points']], hide_index=True, use_container_width=True)


# =========================================================================
# TAB 4: DIAGNOSTICS & SYSTEM LOGS
# =========================================================================
with tab_diagnostics:
    st.markdown("<h3 style='color:#00f2fe;'>System Logs & Optimizer Diagnostics</h3>", unsafe_allow_html=True)
    st.markdown("Inspect raw solver constraints, stdout captures, and Optuna parameter calibrations below.")
    
    # 1. Show Pulp Output log
    st.markdown("#### 🔄 Pulp MIP Solver stdout Log")
    if 'captured_stdout' in locals():
        st.code(captured_stdout, language="text")
    else:
        st.info("Solver has not been executed in this session yet.")
        
    # 2. Show Active configuration details
    st.markdown("#### ⚙️ Active Calibration parameters")
    st.json(pipe_data.get('params', {}))
