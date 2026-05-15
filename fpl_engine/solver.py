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

def _print_gameweek_report(
    t, prob, player_ids, horizon_df, player_details_df, positions,
    objective_column, captain_column,
    squad, starters, captain, trans_in, trans_out,
    free_transfers, paid_transfers,
    WC_WEEK, FH_WEEK, BB_WEEK,
    display_extra_cols=['selected_by_percent','minutes_IDX','fixture_attack_multiplier' , 'fixture_defence_multiplier','GOAL_INDEX','ASSIST_INDEX','CLEAN_SHEET_INDEX','bonus_component','defcon_component',]
):
    """Helper to handle the verbose printing of weekly results."""
    print(f"\n{'='*20} Plan for Gameweek {t} {'='*20}")

    if t == WC_WEEK: print("--- WILDCARD ACTIVE ---")
    if t == FH_WEEK: print("--- FREE HIT ACTIVE ---")
    if t == BB_WEEK: print("--- BENCH BOOST ACTIVE ---")

    # Get weekly scores
    weekly_stats = horizon_df[horizon_df['gameweek'] == t].set_index('id_player')
    weekly_squad_scores = weekly_stats[objective_column]
    weekly_cap_scores = weekly_stats[captain_column]

    # Identify Moves
    trans_in_ids = [p for p in player_ids if (trans_in[p][t].varValue or 0) > 0.9]
    trans_out_ids = [p for p in player_ids if (trans_out[p][t].varValue or 0) > 0.9]
    hits = pulp.value(paid_transfers[t]) * 4

    print(f"Free Transfers Available: {free_transfers[t].varValue:.0f}")
    print(f"Transfers Made: {len(trans_in_ids)} (Hits Cost: {hits:.0f})")

    # --- NEW: setup Display Configuration ---
    display_cols = ['id_player', 'web_name', 'team_name', 'position', 'now_cost', objective_column]

    # Check which extra columns actually exist in the data
    valid_extras = [c for c in display_extra_cols if c in weekly_stats.columns]

    if captain_column != objective_column:
        display_cols.append(captain_column)

    # Add the extra columns  to the list
    display_cols.extend(valid_extras)
    display_cols = list(dict.fromkeys(display_cols))

    # --- Helper to map data to a dataframe ---
    def enrich_df(base_df):
        base_df[objective_column] = base_df.index.map(weekly_squad_scores)
        base_df[captain_column] = base_df.index.map(weekly_cap_scores)
        for col in valid_extras:
            base_df[col] = base_df.index.map(weekly_stats[col])
        return base_df

    # Print Transfers
    if trans_in_ids:
        print("\n--> OUT:")
        out_df = player_details_df.loc[trans_out_ids].copy()
        out_df = enrich_df(out_df) # <--- Apply mapping
        print(out_df.reset_index()[display_cols].round(3).to_string(index=False))

        print("\n<-- IN:")
        in_df = player_details_df.loc[trans_in_ids].copy()
        in_df = enrich_df(in_df) # <--- Apply mapping
        print(in_df.reset_index()[display_cols].round(3).to_string(index=False))

    # Identify Roles
    squad_ids = [p for p in player_ids if squad[p][t].varValue > 0.9]
    starter_ids = [p for p in player_ids if starters[p][t].varValue > 0.9]
    bench_ids = list(set(squad_ids) - set(starter_ids))

    captain_id = max(player_ids, key=lambda p: captain[p][t].varValue or 0)

    # Prepare Starter DataFrame
    start_df = player_details_df.loc[starter_ids].copy()
    start_df = enrich_df(start_df) # <--- Apply mapping

    # Handle Captain Display
    start_df['is_captain'] = 0
    start_df.loc[captain_id, 'is_captain'] = 1
    start_df['web_name'] = np.where(start_df['is_captain'] == 1, start_df['web_name'] + ' (C)', start_df['web_name'])
    start_df['cap_display_score'] = np.where(start_df['is_captain'] == 1, start_df[captain_column], -1)

    # Sorting
    position_order = ['GKP', 'DEF', 'MID', 'FWD']
    start_df['position'] = pd.Categorical(start_df['position'], categories=position_order, ordered=True)

    # Formation Stats
    formation = {pos: sum(1 for p in starter_ids if positions[p] == pos) for pos in position_order}
    print(f"\n--- Starting Formation: {formation['DEF']}-{formation['MID']}-{formation['FWD']} ---")

    print(f"\n--- Starting XI (Ranked by {objective_column}) ---")
    sort_cols = ['cap_display_score',objective_column] + display_extra_cols
    ascending_vals = ([False] * len(sort_cols))
    print(start_df.reset_index().sort_values(sort_cols, ascending=ascending_vals)[display_cols].round(3).to_string(index=False))

    # Prepare Bench DataFrame
    print("\n--- Bench ---")
    bench_df = player_details_df.loc[bench_ids].copy()
    bench_df = enrich_df(bench_df)
    bench_df['position'] = pd.Categorical(bench_df['position'], categories=position_order, ordered=True)
    print(bench_df.reset_index().sort_values(objective_column, ascending=([False] * 1))[display_cols].round(3).to_string(index=False))

    print(f"\n Optimized Squad ID: {[int(x) for x in squad_ids]}")

    # Cost
    total_cost = start_df['now_cost'].sum() + bench_df['now_cost'].sum()
    print(f"\nTotal Squad Cost: £{total_cost:.1f}m")

# --- CELL 40 ---
def plan_sequential_transfers(
    gw_projection_df,
    start_gameweek,
    current_team_ids=None,
    current_realizable_value_dict=None,
    bank_balance=100.0,
    planning_horizon=3,
    initial_free_transfers=1,
    WC_WEEK=None,
    FH_WEEK=None,
    BB_WEEK=None,
    TC_WEEK=None,
    fixed_player_dict=None,
    banned_player_dict=None,
    preferred_teams_dict=None,
    bench_factor=1e-9,
    ft_value=1.5,
    pos_df=None,
    objective_column='custom_score',
    captain_column='ceiling_score',
    bank_aversion=0.01,
    return_model=False
):
    print(f"\n--- Running Sequential Transfer Planner for GW{start_gameweek} ---")

    # =========================================================================
    # 1. SETUP & DATA
    # =========================================================================
    gameweeks = list(range(start_gameweek, np.minimum(start_gameweek + planning_horizon, 39)))

    horizon_df = gw_projection_df[gw_projection_df['gameweek'].isin(gameweeks)].copy()
    player_ids = sorted(horizon_df['id_player'].unique())

    current_team_ids = current_team_ids or []
    current_realizable_value_dict = current_realizable_value_dict or {}
    is_new_season = len(current_team_ids) == 0

    valid_starting_team = [p for p in current_team_ids if p in player_ids]

    buy_costs = horizon_df.groupby('id_player')['now_cost'].first().to_dict()

    spread_dict = {}
    for p in player_ids:
        if p in valid_starting_team:
            realizable = current_realizable_value_dict.get(p, buy_costs[p])
            spread_dict[p] = buy_costs[p] - realizable
        else:
            spread_dict[p] = 0.0

    positions  = horizon_df.groupby('id_player')['position'].first().to_dict()
    teams      = horizon_df.groupby('id_player')['team_name'].first().to_dict()
    points_safe = horizon_df.set_index(['id_player', 'gameweek'])[objective_column].to_dict()
    points_cap  = horizon_df.set_index(['id_player', 'gameweek'])[captain_column].to_dict()
    # Read the differential discount from the dataframe
    bonus_mult = horizon_df.set_index(['id_player', 'gameweek'])['raw_bonus_multiplier'].to_dict()

    if pos_df is None:
        pos_df = pd.DataFrame([
            {'singular_name_short': 'GKP', 'squad_select': 2, 'squad_min_play': 1, 'squad_max_play': 1},
            {'singular_name_short': 'DEF', 'squad_select': 5, 'squad_min_play': 3, 'squad_max_play': 5},
            {'singular_name_short': 'MID', 'squad_select': 5, 'squad_min_play': 2, 'squad_max_play': 5},
            {'singular_name_short': 'FWD', 'squad_select': 3, 'squad_min_play': 1, 'squad_max_play': 3},
        ])

    # =========================================================================
    # 2. VARIABLES
    # =========================================================================
    prob = pulp.LpProblem("FPL_Sequential_Model", pulp.LpMaximize)

    squad    = pulp.LpVariable.dicts("Squad",   (player_ids, gameweeks), cat='Binary')
    starters = pulp.LpVariable.dicts("Starter", (player_ids, gameweeks), cat='Binary')
    captain  = pulp.LpVariable.dicts("Captain", (player_ids, gameweeks), cat='Binary')

    trans_in  = pulp.LpVariable.dicts("TransIn",  (player_ids, gameweeks), cat='Binary')
    trans_out = pulp.LpVariable.dicts("TransOut", (player_ids, gameweeks), cat='Binary')

    cont_hold = pulp.LpVariable.dicts("ContHold", (player_ids, gameweeks), cat='Binary')

    # Tracks which specific incoming transfers triggered a -4 hit
    paid_trans_in = pulp.LpVariable.dicts("PaidTransIn", (player_ids, gameweeks),cat='Binary')

    bank = pulp.LpVariable.dicts("Bank", gameweeks, lowBound=0)

    transfers_made = pulp.LpVariable.dicts("TransfersMade", gameweeks, lowBound=0)
    paid_transfers = pulp.LpVariable.dicts("PaidTransfers", gameweeks, lowBound=0, cat='Integer')
    free_transfers = pulp.LpVariable.dicts("FreeTransfers", gameweeks, lowBound=1, upBound=5, cat='Integer')
    ft_used        = pulp.LpVariable.dicts("FT_Used",        gameweeks, lowBound=0, upBound=5, cat='Integer')

    carry_ft = pulp.LpVariable.dicts("CarryFT", (gameweeks, range(1, 6)), cat='Binary')

    # =========================================================================
    # 3. OBJECTIVE
    # =========================================================================
    gw_weight    = {t: 1.0 for t in gameweeks}
    is_tc        = {t: 1 if t == TC_WEEK else 0 for t in gameweeks}
    bench_weights = {t: 1.0 if t == BB_WEEK else bench_factor for t in gameweeks}

    transfer_friction = 0.05

    squad_score = pulp.lpSum(
        gw_weight[t] * (
            starters[p][t] * points_safe.get((p, t), 0)
            + captain[p][t] * (1.0 + is_tc[t]) * points_cap.get((p, t), 0)
            + (squad[p][t] - starters[p][t]) * bench_weights[t] * points_safe.get((p, t), 0)
        )
        for p in player_ids for t in gameweeks
    )

    carry_ft_reward = pulp.lpSum(
        gw_weight[t] * pulp.lpSum(
            carry_ft[t][n] * (ft_value * ((len(gameweeks) - i) / planning_horizon) / n)
            for n in range(1, 6)
        )
        for i, t in enumerate(gameweeks[:-1])
    )

    transfer_costs = (
        pulp.lpSum(4 * paid_transfers[t] for t in gameweeks)
        + pulp.lpSum(transfer_friction * transfers_made[t] for t in gameweeks)
    )

    # Calculate the rebate: 4 points * the player's specific discount multiplier
    transfer_hit_discount = pulp.lpSum(
        paid_trans_in[p][t] * 4.0 * bonus_mult.get((p, t), 0.0)
        for p in player_ids for t in gameweeks
    )

    bank_penalty = pulp.lpSum(bank_aversion * bank[t] for t in gameweeks)

    # --- G. Starting XI Differential Tie-Breaker ---
    # We apply a micro-weight to the bonus multiplier ONLY for starters.
    # Max bonus is 0.15. Multiplied by 0.5, the absolute maximum "fake" points
    # added here is 0.075. This is strictly a tie-breaker and will never be
    # large enough to trigger a -4 hit (which requires 4.0 points of utility).
    tiebreaker_weight = 0.5

    starter_diff_tiebreaker = pulp.lpSum(
        starters[p][t] * bonus_mult.get((p, t), 0.0) * tiebreaker_weight
        for p in player_ids for t in gameweeks
    )

    # Update final objective function to include the discount
    # Update final objective function to include the tie-breaker
    prob += squad_score + carry_ft_reward - transfer_costs + transfer_hit_discount - bank_penalty + starter_diff_tiebreaker

    # =========================================================================
    # 4. CONSTRAINTS
    # =========================================================================
    for i, t in enumerate(gameweeks):
        prev_t  = gameweeks[i - 1] if i > 0 else None
        is_wc   = (t == WC_WEEK)
        is_fh   = (t == FH_WEEK)
        is_gw1  = (t == start_gameweek)

        # --- A. Squad Composition ---
        prob += pulp.lpSum(squad[p][t]   for p in player_ids) == 15
        prob += pulp.lpSum(starters[p][t] for p in player_ids) == 11
        prob += pulp.lpSum(captain[p][t]  for p in player_ids) == 1

        for _, r in pos_df.iterrows():
            pos_players = [p for p in player_ids if positions[p] == r['singular_name_short']]
            prob += pulp.lpSum(squad[p][t]    for p in pos_players) == r['squad_select']
            prob += pulp.lpSum(starters[p][t] for p in pos_players) >= r['squad_min_play']
            prob += pulp.lpSum(starters[p][t] for p in pos_players) <= r['squad_max_play']

        for team in horizon_df['team_name'].unique():
            prob += pulp.lpSum(squad[p][t] for p in player_ids if teams[p] == team) <= 3

        for p in player_ids:
            prob += starters[p][t] <= squad[p][t]
            prob += captain[p][t]  <= starters[p][t]

        # --- B. State Transitions & Transfers ---
        for p in player_ids:
            if is_gw1:
                if is_new_season:
                    prob += trans_out[p][t] == 0
                    prob += squad[p][t] == trans_in[p][t]
                    prob += cont_hold[p][t] == 0
                else:
                    init = 1 if p in valid_starting_team else 0
                    prob += squad[p][t] == init - trans_out[p][t] + trans_in[p][t]
                    prob += cont_hold[p][t] == (init - trans_out[p][t])
                    prob += trans_out[p][t] <= init
                    prob += trans_in[p][t]  <= 1 - init

            elif is_fh:
                prob += squad[p][t] == squad[p][prev_t] - trans_out[p][t] + trans_in[p][t]
                prob += cont_hold[p][t] <= cont_hold[p][prev_t]
                prob += cont_hold[p][t] <= squad[p][t]

            elif prev_t == FH_WEEK:
                if i < 2:
                    anchor_squad = 1 if p in valid_starting_team else 0
                    anchor_hold  = 1 if p in valid_starting_team else 0
                else:
                    anchor_gw    = gameweeks[i - 2]
                    anchor_squad = squad[p][anchor_gw]
                    anchor_hold  = cont_hold[p][anchor_gw]

                prob += squad[p][t] == anchor_squad - trans_out[p][t] + trans_in[p][t]
                prob += cont_hold[p][t] <= anchor_hold
                prob += cont_hold[p][t] <= anchor_hold - trans_out[p][t]
                prob += cont_hold[p][t] <= squad[p][t]

            else:
                prob += squad[p][t] == squad[p][prev_t] - trans_out[p][t] + trans_in[p][t]
                prob += trans_out[p][t] <= squad[p][prev_t]
                prob += trans_in[p][t]  <= 1 - squad[p][prev_t]
                prob += cont_hold[p][t] <= cont_hold[p][prev_t]
                prob += cont_hold[p][t] <= squad[p][t]

        # --- C. Financial Logic ---
        cost_in = pulp.lpSum(trans_in[p][t] * buy_costs[p] for p in player_ids)

        if is_gw1:
            if is_new_season:
                prob += bank[t] == bank_balance - cost_in
            else:
                spread_correction = pulp.lpSum(
                    trans_out[p][t] * spread_dict[p]
                    for p in valid_starting_team
                )
                revenue_out = pulp.lpSum(trans_out[p][t] * buy_costs[p] for p in player_ids) - spread_correction
                prob += bank[t] == bank_balance + revenue_out - cost_in
        else:
            spread_correction = pulp.lpSum(
                (cont_hold[p][prev_t] - cont_hold[p][t]) * spread_dict[p]
                for p in player_ids
            )
            revenue_out = pulp.lpSum(trans_out[p][t] * buy_costs[p] for p in player_ids) - spread_correction
            prob += bank[t] == bank[prev_t] + revenue_out - cost_in

        prob += bank[t] >= 0

        # --- D. Free Transfer Accounting ---
        prob += transfers_made[t] == pulp.lpSum(trans_in[p][t] for p in player_ids)

        if is_gw1:
            if is_new_season:
                prob += free_transfers[t] == 1
                prob += paid_transfers[t] == 0
                prob += ft_used[t] == 1
            else:
                prob += free_transfers[t] == initial_free_transfers
                prob += ft_used[t] <= free_transfers[t]
                prob += ft_used[t] <= transfers_made[t]
                if is_wc or is_fh:
                    prob += ft_used[t] == 1
                    prob += paid_transfers[t] == 0
                else:
                    prob += paid_transfers[t] == transfers_made[t] - ft_used[t]
        else:
            prob += free_transfers[t] <= (free_transfers[prev_t] - ft_used[prev_t]) + 1
            prob += free_transfers[t] <= 5
            prob += free_transfers[t] >= 1

            prob += ft_used[t] <= free_transfers[t]
            prob += ft_used[t] <= transfers_made[t]

            if is_wc or is_fh:
                prob += ft_used[t]        == 1
                prob += paid_transfers[t] == 0
            else:
                prob += paid_transfers[t] == transfers_made[t] - ft_used[t]

        remaining_ft = free_transfers[t] - ft_used[t]
        for n in range(1, 6):
            prob += remaining_ft >= n * carry_ft[t][n]

        # --- E. Paid Transfer Discount Accounting ---
        for p in player_ids:
            # A player can only be a 'paid transfer in' if they were actually transferred in
            prob += paid_trans_in[p][t] <= trans_in[p][t]

        # The sum of all 'paid transfers in' must exactly equal the number of hits taken that week
        prob += pulp.lpSum(paid_trans_in[p][t] for p in player_ids) == paid_transfers[t]

        # --- F. User Constraints ---
        if fixed_player_dict:
            for p in list(fixed_player_dict.get('Default', [])) + list(fixed_player_dict.get(t, [])):
                if p in player_ids:
                    prob += squad[p][t] == 1

        if banned_player_dict:
            for p in list(banned_player_dict.get('Default', [])) + list(banned_player_dict.get(t, [])):
                if p in player_ids:
                    prob += squad[p][t] == 0

    # =========================================================================
    # 5. SOLVE & OUTPUT
    # =========================================================================
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=180))

    if pulp.LpStatus[prob.status] == 'Optimal':
        print("\n" + "=" * 60)
        print("Optimal Sequential Transfer Plan Found!")
        print(f"Projected Total Utility: {pulp.value(prob.objective):.2f}")
        print("=" * 60)

        player_details_df = gw_projection_df.drop_duplicates('id_player').set_index('id_player')

        for t in gameweeks:
            _print_gameweek_report(
                t, prob, player_ids, horizon_df, player_details_df, positions,
                objective_column, captain_column,
                squad, starters, captain, trans_in, trans_out,
                free_transfers, paid_transfers,
                WC_WEEK, FH_WEEK, BB_WEEK,
            )

        if return_model:
            return prob, {
                "player_ids":    player_ids,
                "squad":         squad,
                "starters":      starters,
                "captain":       captain,
                "trans_in":      trans_in,
                "paid_transfers": paid_transfers,
            }
    else:
        print("Optimization Failed. Check constraint conflicts.")
        return None

# --- CELL 41 ---
def create_optimal_fpl_team(budget = 100):
    """
    Selects an optimal fantasy football team (11 starters, 4 bench) based on
    player data using linear programming, visualizes team points by position,
    analyzes team cost, and returns the starter and bench DataFrames.

    Args:
        budget: The maximum allowed total cost for the squad.

    Returns:
        A tuple containing two pandas DataFrames:
        - starter_df_full: DataFrame for the starting 11 players.
        - bench_df_full: DataFrame for the 4 bench players.
    """
    player_df = get_current_players_df()
    # Linear Programming Optimization
    player_ids = sorted(player_df['id'].unique())
    points = player_df.set_index(['id'])['total_points'].to_dict()
    costs = player_df.set_index(['id'])['now_cost'].to_dict()
    positions = player_df.set_index(['id'])['position'].to_dict()
    teams = player_df.set_index(['id'])['team_name'].to_dict()
    names = player_df.set_index(['id'])['web_name'].to_dict()

    prob = pulp.LpProblem("FPL_Dreamteam", pulp.LpMaximize)

    squad = pulp.LpVariable.dicts("Squad", (player_ids), cat='Binary')
    starters = pulp.LpVariable.dicts("Starter", (player_ids), cat='Binary')

    # Objective function: Maximize total points (including bench points at 5%)
    total_score = pulp.lpSum(starters[p] * points.get(p, 0) for p in player_ids) + pulp.lpSum(
                    (squad[p] - starters[p]) * points.get(p, 0) * 0.05
                    for p in player_ids
                )
    prob += total_score, "Total Score"

    # Constraints
    total_cost = pulp.lpSum(squad[p] * costs.get(p, 0) for p in player_ids)
    prob += total_cost <= budget, "Total Cost Constraint"

    prob += pulp.lpSum(starters[p] for p in player_ids) == 11, "Number of Starters"
    prob += pulp.lpSum(squad[p] for p in player_ids) == 15, "Squad Size"

    # Position constraints for squad and starters
    pos_constraint = get_pos_constraint_df()
    for _, row in pos_constraint.iterrows():
        pos = row['singular_name_short']
        min_players = row['squad_min_play']
        max_players = row['squad_max_play']
        squad_select = row['squad_select']

        prob += pulp.lpSum(squad[p] for p in player_ids if positions.get(p) == pos) == squad_select, f"Squad {pos} Constraint"

        if pd.notna(min_players):
            prob += pulp.lpSum(starters[p] for p in player_ids if positions.get(p) == pos) >= min_players, f"Starter Min {pos} Constraint"
        if pd.notna(max_players):
            prob += pulp.lpSum(starters[p] for p in player_ids if positions.get(p) == pos) <= max_players, f"Starter Max {pos} Constraint"

    # Team constraints
    for team in player_df['team_name'].unique():
        prob += pulp.lpSum(squad[p] for p in player_ids if teams.get(p) == team) <= 3, f"Max 3 players from {team}"

    # Ensure starters are also in the squad
    for p in player_ids:
        prob += starters[p] <= squad[p], f"Starter {p} must be in Squad"

    # Solve the problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=180))

    # Team Selection and Sorting
    starter_ids = [p for p in player_ids if pulp.value(starters[p]) == 1]
    bench_ids = [p for p in player_ids if pulp.value(squad[p]) == 1 and pulp.value(starters[p]) == 0]

    starter_df_full = player_df[player_df['id'].isin(starter_ids)].copy()
    bench_df_full = player_df[player_df['id'].isin(bench_ids)].copy()

    position_order = ['GKP', 'DEF', 'MID', 'FWD']
    starter_df_full['position'] = pd.Categorical(starter_df_full['position'], categories=position_order, ordered=True)
    bench_df_full['position'] = pd.Categorical(bench_df_full['position'], categories=position_order, ordered=True)

    starter_df_full = starter_df_full.sort_values(by=['position', 'total_points'], ascending=[True, False])
    bench_df_full = bench_df_full.sort_values(by=['position', 'total_points'], ascending=[True, False])

    # Visualization
    print(f"")
    print(20*"#"+f" Starter "+20*"#")
    print(f"")
    print(starter_df_full[
        [
            'id' ,
            'web_name' ,
            'now_cost' ,
            'position' ,
            'team_name' ,
            'selected_by_percent',
            'total_points',
        ]
    ].to_string(index = False))
    print(f"")
    print(20*"#"+f" Bench "+20*"#")
    print(f"")
    print(bench_df_full[
        [
            'id' ,
            'web_name' ,
            'now_cost' ,
            'position' ,
            'team_name' ,
            'selected_by_percent',
            'total_points',
        ]
    ].to_string(index = False))

    # Cost Analysis
    starter_cost = starter_df_full['now_cost'].sum()
    bench_cost = bench_df_full['now_cost'].sum()
    total_squad_cost = starter_cost + bench_cost


    print(f"")
    print(f"Total cost of the entire squad: £{total_squad_cost:.2f}")
    print(f"Total cost of the starting team: £{starter_cost:.2f}")
    print(f"Total cost of the bench: £{bench_cost:.2f}")

    if total_squad_cost <= budget:
        print(f"The total squad cost (£{total_squad_cost:.2f}) is within the budget (£{budget:.2f}).")
    else:
        print(f"The total squad cost (£{total_squad_cost:.2f}) exceeds the budget (£{budget:.2f}).")

    return starter_df_full, bench_df_full

# --- CELL 42 ---
def optimize_players_by_idx(df, budget, num_slots, position, fixed_players=None, banned_players=None, preferred_teams_dict=None, bench_factor=0.05, pos_df = None, objective_column='Perf_IDX'):
    """
    Optimizes player selection based on their calculated IDX score for a specific position
    or across all positions, within budget and slot constraints, with support for fixed and banned players.

    Args:
        df (pd.DataFrame): DataFrame containing player data with IDX scores (top_player_stats).
        budget (float): Maximum total cost for the selected players.
        num_slots (int or dict): Number of players to select. If 'position' is a single position,
                                 this should be an integer. If 'position' is 'all', this should
                                 be a dictionary where keys are positions ('GKP', 'DEF', 'MID', 'FWD')
                                 and values are the number of slots for each position (total 15).
        position (str): The position to optimize for ('GKP', 'DEF', 'MID', 'FWD', or 'all').
        fixed_players (list, optional): A list of player IDs that must be included in the selection.
                                        Defaults to None.
        banned_players (list, optional): A list of player IDs that must NOT be included in the selection.
                                         Defaults to None.
        preferred_teams_dict (dict, optional): A dictionary where keys are team names and values
                                               are the minimum number of players required from that team.
                                               Defaults to None.
        bench_factor (float, optional): The factor by which bench players' objective_column is multiplied in the objective. Defaults to 0.05.
        pos_df (pd.DataFrame, optional): DataFrame with position constraints (squad_select, squad_min_play, squad_max_play). Defaults to get_pos_constraint_df().
        objective_column (str, optional): The column to use for optimization. Defaults to 'Perf_IDX'.

    Returns:
        pd.DataFrame: DataFrame of selected players, or None if no optimal solution is found.
    """
    if fixed_players is None:
        fixed_players = []
    if banned_players is None:
        banned_players = []
    if preferred_teams_dict is None:
        preferred_teams_dict = {}
    if pos_df is None:
        pos_df = get_pos_constraint_df()


    all_positions = ['GKP', 'DEF', 'MID', 'FWD']
    if position != 'all' and position not in all_positions:
        print(f"Invalid position: {position}. Please use 'GKP', 'DEF', 'MID', 'FWD', or 'all'.")
        return None

    if position == 'all' and not isinstance(num_slots, dict):
        print("Error: When 'position' is 'all', 'num_slots' must be a dictionary specifying slots per position.")
        return None
    elif position != 'all' and not isinstance(num_slots, int):
         print(f"Error: When 'position' is '{position}', 'num_slots' must be an integer.")
         return None

    if objective_column not in df.columns:
         raise ValueError(f"Objective column '{objective_column}' not found in the input DataFrame. Please check the column name.")


    if position == 'all':
        # Ensure num_slots sums to 15 for a full squad optimization
        if not (isinstance(num_slots, dict) and sum(num_slots.values()) == 15):
             print("Error: When 'position' is 'all', 'num_slots' must be a dictionary specifying slots for a 15-player squad.")
             return None

        # Initialize combined_selected_players with necessary columns
        required_cols_all = ['id_player', 'now_cost', 'team_name', 'web_name', 'position', objective_column] + [f'{pos}_IDX' for pos in all_positions]
        # Ensure all required columns exist in the input df before selecting
        required_cols_all = [col for col in required_cols_all if col in df.columns]

        # Filter and add fixed players directly to the combined_selected_players list for later inclusion
        fixed_players_in_df = df[df['id_player'].isin(fixed_players)].copy()
        # Add 'Is_Starter' column to fixed players, initialized to 0 (bench)
        # Fixed players are not initially assumed to be starters; their starting status is determined by optimization
        fixed_players_in_df['Is_Starter'] = 0

        # Create a pool of players to optimize (excluding fixed and banned players)
        players_to_optimize_df = df[~df['id_player'].isin(fixed_players_in_df['id_player'].tolist() + banned_players)].copy()

        # Handle potential NaN/inf values in the objective_column for optimization
        players_to_optimize_df[objective_column] = players_to_optimize_df[objective_column].replace([np.inf, -np.inf], np.nan).fillna(0)
        fixed_players_in_df[objective_column] = fixed_players_in_df[objective_column].replace([np.inf, -np.inf], np.nan).fillna(0)


        if players_to_optimize_df.empty and len(fixed_players_in_df) < 15:
            print("No players available to optimize after excluding fixed and banned players, but slots remain.")
            # Return only the fixed players if no optimization is possible but fixed players exist
            if not fixed_players_in_df.empty:
                 # Sort by position order and then objective_column descending
                position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
                fixed_players_in_df['position_order'] = fixed_players_in_df['position'].map(position_order)
                fixed_players_in_df = fixed_players_in_df.sort_values(by=['Is_Starter','position_order', objective_column], ascending=[ False,True,False]).drop(columns='position_order')
                print(f"Returning fixed players. Total Cost: £{fixed_players_in_df['now_cost'].sum():.1f}m")
                return fixed_players_in_df[['id_player', 'web_name', 'team_name', 'now_cost', 'position', 'Is_Starter', objective_column,'minutes_IDX',]].round(3)
            else:
                 return None


        # Prepare data for the single optimization problem
        players_to_optimize = players_to_optimize_df['id_player'].tolist()
        costs = dict(zip(players_to_optimize_df['id_player'], players_to_optimize_df['now_cost']))
        objective_scores = dict(zip(players_to_optimize_df['id_player'], players_to_optimize_df[objective_column]))
        positions_dict = dict(zip(players_to_optimize_df['id_player'], players_to_optimize_df['position']))
        teams_dict = dict(zip(players_to_optimize_df['id_player'], players_to_optimize_df['team_name']))

        prob = pulp.LpProblem(f"Optimize_All_Players_by_{objective_column}", pulp.LpMaximize)

        # Decision variables
        # x[p] is 1 if player p is selected in the SQUAD (for players to optimize)
        x = pulp.LpVariable.dicts("Select", players_to_optimize, cat='Binary')
        # y[p] is 1 if player p is in the STARTING XI (for players to optimize)
        y = pulp.LpVariable.dicts("Starter", players_to_optimize, cat='Binary')

        # Total weighted score from both fixed and optimized players
        # Fixed players are assumed to be in the squad; their starter status is determined below
        fixed_players_squad = pulp.LpVariable.dicts("FixedSquad", fixed_players_in_df['id_player'].tolist(), cat='Binary')
        fixed_players_starters = pulp.LpVariable.dicts("FixedStarter", fixed_players_in_df['id_player'].tolist(), cat='Binary')

        # Constraint: Fixed players must be in the squad
        for p_id in fixed_players_in_df['id_player'].tolist():
             prob += fixed_players_squad[p_id] == 1

        # Constraint: Fixed starters must be in the fixed squad
        for p_id in fixed_players_in_df['id_player'].tolist():
             prob += fixed_players_starters[p_id] <= fixed_players_squad[p_id] # This is always true since fixed_players_squad is 1

        # Objective function: Maximize the sum of objective_column for starters + bench_factor * objective_column for bench
        # This includes both fixed and optimized players
        fixed_scores = dict(zip(fixed_players_in_df['id_player'], fixed_players_in_df[objective_column]))

        prob += pulp.lpSum(fixed_scores[p] * fixed_players_starters[p] for p in fixed_players_in_df['id_player'].tolist()) + \
                pulp.lpSum(fixed_scores[p] * bench_factor * (fixed_players_squad[p] - fixed_players_starters[p]) for p in fixed_players_in_df['id_player'].tolist()) + \
                pulp.lpSum(objective_scores[p] * y[p] for p in players_to_optimize) + \
                pulp.lpSum(objective_scores[p] * bench_factor * (x[p] - y[p]) for p in players_to_optimize), f"Total_Weighted_{objective_column}_Score_Optimized"


        # Constraints
        # 1. Total cost constraint (for *all* players in the squad)
        fixed_costs = dict(zip(fixed_players_in_df['id_player'], fixed_players_in_df['now_cost']))
        prob += pulp.lpSum(costs[p] * x[p] for p in players_to_optimize) + pulp.lpSum(fixed_costs[p_id] * fixed_players_squad[p_id] for p_id in fixed_players_in_df['id_player'].tolist()) <= budget, "Total_Budget"

         # 2. Total number of players (squad size must be 15)
        prob += pulp.lpSum(x[p] for p in players_to_optimize) + pulp.lpSum(fixed_players_squad[p_id] for p_id in fixed_players_in_df['id_player'].tolist()) == 15, "Total_Squad_Size"

        # 3. Position constraints for the *squad* (including fixed players)
        fixed_squad_positions = dict(zip(fixed_players_in_df['id_player'], fixed_players_in_df['position']))
        for pos, count in num_slots.items():
            prob += pulp.lpSum(x[p] for p in players_to_optimize if positions_dict.get(p) == pos) + \
                    pulp.lpSum(fixed_players_squad[p_id] for p_id in fixed_players_in_df['id_player'].tolist() if fixed_squad_positions.get(p_id) == pos) == count, f"Squad_Slots_{pos}"

        # 4. Team constraint (Max 3 players per team in the squad)
        # Need to consider teams of fixed players as well
        fixed_squad_teams = dict(zip(fixed_players_in_df['id_player'], fixed_players_in_df['team_name']))
        # Get the 'team_name' column from each DataFrame as a NumPy array
        teams1 = players_to_optimize_df['team_name'].values
        teams2 = fixed_players_in_df['team_name'].values

        # Concatenate the two arrays into one
        all_teams_array = np.concatenate([teams1, teams2])

        # Get the unique values from the combined array
        unique_teams = np.unique(all_teams_array)
        for team_name in unique_teams:
             prob += pulp.lpSum(x[p] for p in players_to_optimize if teams_dict.get(p) == team_name) + \
                     pulp.lpSum(fixed_players_squad[p_id] for p_id in fixed_players_in_df['id_player'].tolist() if fixed_squad_teams.get(p_id) == team_name) <= 3, f"Max_3_Players_{team_name}"

        # 5. Preferred Team Constraints
        if preferred_teams_dict:
            for team_name, min_count in preferred_teams_dict.items():
                 prob += pulp.lpSum(x[p] for p in players_to_optimize if teams_dict.get(p) == team_name) + \
                         pulp.lpSum(fixed_players_squad[p_id] for p_id in fixed_players_in_df['id_player'].tolist() if fixed_squad_teams.get(p_id) == team_name) >= min_count, f"Min_Players_{team_name}"


        # 6. Starting XI constraints
        prob += pulp.lpSum(y[p] for p in players_to_optimize) + pulp.lpSum(fixed_players_starters[p_id] for p_id in fixed_players_in_df['id_player'].tolist()) == 11, "Total_Starting_XI_Size"
        for p in players_to_optimize:
            prob += y[p] <= x[p] # A starter must be in the squad


        # 7. Positional constraints for the *starting XI* from pos_df (including fixed starters)
        for _, row in pos_df.iterrows():
            pos = row['singular_name_short']
            squad_min_play = row['squad_min_play']
            squad_max_play = row['squad_max_play']

            if pd.notna(squad_min_play):
                prob += pulp.lpSum(y[p] for p in players_to_optimize if positions_dict.get(p) == pos) + \
                        pulp.lpSum(fixed_players_starters[p_id] for p_id in fixed_players_in_df['id_player'].tolist() if fixed_squad_positions.get(p_id) == pos) >= squad_min_play, f"Min_Starters_{pos}"
            if pd.notna(squad_max_play):
                 prob += pulp.lpSum(y[p] for p in players_to_optimize if positions_dict.get(p) == pos) + \
                         pulp.lpSum(fixed_players_starters[p_id] for p_id in fixed_players_in_df['id_player'].tolist() if fixed_squad_positions.get(p_id) == pos) <= squad_max_play, f"Max_Starters_{pos}"


        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Check the solution status
        if pulp.LpStatus[prob.status] == 'Optimal':
            print("\nOptimal selection found for the combined squad!")
            selected_players_ids = [p for p in players_to_optimize if x[p].varValue > 0.9] + fixed_players_in_df['id_player'].tolist()
            selected_players_df = df[df['id_player'].isin(selected_players_ids)].copy() # Use the original df to get all columns

            # Determine starter status for all selected players
            starter_ids = [p for p in players_to_optimize if y[p].varValue > 0.9] + \
                          [p_id for p_id in fixed_players_in_df['id_player'].tolist() if fixed_players_starters[p_id].varValue > 0.9]

            selected_players_df['Is_Starter'] = selected_players_df['id_player'].apply(lambda p_id: 1 if p_id in starter_ids else 0)

            # Handle potential NaN/inf values in objective_column before recalculating score
            selected_players_df[objective_column] = selected_players_df[objective_column].replace([np.inf, -np.inf], np.nan).fillna(0)

            total_cost = selected_players_df['now_cost'].sum()
            # Recalculate total weighted score based on the final squad
            total_weighted_score = selected_players_df.apply(
                lambda row: row[objective_column] if row['Is_Starter'] == 1 else row[objective_column] * bench_factor, axis=1
            ).sum()


            print(f"Total Combined Weighted {objective_column} Score: {total_weighted_score:.2f}")
            print(f"Total Combined Cost: £{total_cost:.1f}m")

            # Sort by position order and then objective_column descending
            position_order = {'GKP': 0, 'DEF': 1, 'MID': 2, 'FWD': 3}
            selected_players_df['position_order'] = selected_players_df['position'].map(position_order)
            selected_players_df = selected_players_df.sort_values(by=['Is_Starter','position_order', objective_column], ascending=[ False,True,False]).drop(columns='position_order')

            print(selected_players_df['id_player'].tolist())
            return selected_players_df[['id_player', 'web_name', 'team_name', 'now_cost', 'position', 'Is_Starter', objective_column,'minutes_IDX',]].round(3)

        else:
            print(f"\nCould not find an optimal solution for the combined squad. Status: {pulp.LpStatus[prob.status]}")
            return None

    else: # Optimize for a single position
        position_df = df[df['position'] == position].copy()

        if position_df.empty:
            print(f"No players found for position: {position}.")
            return None

        idx_column = objective_column # Use the specified objective_column

        required_cols = ['id_player', 'now_cost', idx_column]
        for col in required_cols:
            if col not in position_df.columns:
                print(f"Error: Required column '{col}' not found in the DataFrame.")
                return None
            if col != 'id_player':
                position_df[col] = pd.to_numeric(position_df[col], errors='coerce')

        position_df.dropna(subset=required_cols, inplace=True)

        # Handle potential NaN/inf values in the IDX column for optimization
        position_df[idx_column] = position_df[idx_column].replace([np.inf, -np.inf], np.nan).fillna(0)

        if position_df.empty:
            print(f"No valid player data after cleaning for position: {position}.")
            return None

        # Filter out banned players
        position_df = position_df[~position_df['id_player'].isin(banned_players)].copy()

        # Ensure fixed players are in the filtered DataFrame and update num_slots and budget
        initial_cost = 0
        initial_slots = 0
        fixed_players_in_pos = [p_id for p_id in fixed_players if p_id in position_df['id_player'].tolist()]

        for p_id in fixed_players_in_pos:
            player_row = position_df[position_df['id_player'] == p_id].iloc[0]
            initial_cost += player_row['now_cost']
            initial_slots += 1

        adjusted_budget = budget - initial_cost
        adjusted_num_slots = num_slots - initial_slots

        if adjusted_num_slots < 0:
            print(f"Error: The number of fixed players ({initial_slots}) exceeds the required slots ({num_slots}) for position {position}.")
            return None
        if adjusted_budget < 0:
            print(f"Error: The cost of fixed players (£{initial_cost:.1f}m) exceeds the budget (£{budget:.1f}m) for position {position}.")
            return None


        # Create a dictionary of players and their IDX scores and costs (excluding fixed players for optimization)
        players_to_optimize = position_df[~position_df['id_player'].isin(fixed_players_in_pos)]['id_player'].tolist()
        costs = dict(zip(position_df['id_player'], position_df['now_cost']))
        objective_scores = dict(zip(position_df['id_player'], position_df[idx_column]))
        teams_dict = dict(zip(position_df['id_player'], position_df['team_name'])) # Added for preferred teams

        # Create the LP problem
        prob = pulp.LpProblem(f"Optimize_{position}_Players_by_{objective_column}", pulp.LpMaximize)

        # Decision variables: x[p] is 1 if player p is selected, 0 otherwise
        x = pulp.LpVariable.dicts("Select", players_to_optimize, cat='Binary')

        # Objective function: Maximize the sum of IDX scores of selected players
        # This only considers the players being optimized; fixed players are implicitly included
        prob += pulp.lpSum(objective_scores[p] * x[p] for p in players_to_optimize), f"Total_{position}_{objective_column}_Score_Optimized"

        # Constraints
        # 1. Total cost constraint (for players to optimize)
        prob += pulp.lpSum(costs[p] * x[p] for p in players_to_optimize) <= adjusted_budget, "Remaining_Budget"

        # 2. Number of players constraint (for players to optimize)
        prob += pulp.lpSum(x[p] for p in players_to_optimize) == adjusted_num_slots, "Remaining_Slots"

        # 3. Preferred Team Constraints (only for players being optimized within this position)
        if preferred_teams_dict:
            for team_name, min_count in preferred_teams_dict.items():
                 # Only apply the constraint if the team is relevant to the current position
                 if team_name in position_df['team_name'].unique():
                     # Need to consider fixed players in this position for the count
                     fixed_players_from_team = [p_id for p_id in fixed_players_in_pos if teams_dict.get(p_id) == team_name]
                     prob += pulp.lpSum(x[p] for p in players_to_optimize if teams_dict.get(p) == team_name) + len(fixed_players_from_team) >= min_count, f"Min_Players_{team_name}_{position}"


        # Solve the problem
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        # Check the solution status
        if pulp.LpStatus[prob.status] == 'Optimal':
            print(f"\nOptimal selection found for {position}!")
            selected_players_ids = [p for p in players_to_optimize if x[p].varValue > 0.9] + fixed_players_in_pos
            selected_players_df = position_df[position_df['id_player'].isin(selected_players_ids)].copy()
            selected_players_df = selected_players_df.sort_values(by=idx_column, ascending=False).reset_index(drop=True)

            total_cost = selected_players_df['now_cost'].sum()
            total_idx_score = selected_players_df[idx_column].sum()

            print(f"Total {position} {objective_column} Score: {total_idx_score:.2f}")
            print(f"Total Cost: £{total_cost:.1f}m")

            return selected_players_df[['id_player', 'web_name', 'team_name', 'now_cost', idx_column,'minutes_IDX',]].round(3)
        else:
            print(f"\nCould not find an optimal solution for {position}. Status: {pulp.LP_STATUS[prob.status]}")
            return None

# --- CELL 44 ---
