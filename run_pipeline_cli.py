import asyncio
import sys
import os
import pandas as pd
import numpy as np
import pulp

# Add workspace directory to path to ensure clean imports
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app import run_fpl_pipeline
from fpl_engine.config import MY_FPL_ID
from fpl_engine.scoring import create_optimized_custom_score
from fpl_engine.data import get_my_player_ids
from fpl_engine.solver import plan_sequential_transfers

async def main():
    print("==============================================================")
    print("⚽ FPL SQUAD OPTIMIZER & TRANSFER SOLVER (CLI RUNNER) ⚽")
    print("==============================================================")
    
    manager_id = MY_FPL_ID
    print(f"🤖 Target FPL Manager ID: {manager_id}")
    
    # 1. Execute the full calculation pipeline
    print("\n⏳ Executing FPL projection calculation pipeline...")
    pipe_data = await run_fpl_pipeline(manager_id)
    current_gw = pipe_data['current_gw']
    players_df = pipe_data['players_df']
    gw_projection_df = pipe_data['gw_projection_df']
    default_weights = pipe_data['weights']
    params = pipe_data['params']
    
    print(f"✅ Pipeline Loaded successfully!")
    print(f"   - Current Gameweek: GW {current_gw}")
    print(f"   - Loaded Players count: {len(players_df)}")
    print(f"   - Total Gameweek Projections: {len(gw_projection_df)}")
    print(f"   - Solver Weights Mode: {default_weights.get('mode', 'N/A')}")
    
    if gw_projection_df.empty:
        print("❌ Error: Projections DataFrame is empty. Solver cannot continue.")
        return
        
    # 2. Calibrate custom performance scores
    print("\n⚖️ Calibrating player custom performance scores...")
    diff_weight = default_weights.get('diff_weight', 0.13)
    upside_weight = default_weights.get('upside_weight', 0.12)
    gw_projection_df = create_optimized_custom_score(
        df=gw_projection_df,
        differential_weight=diff_weight,
        upside_weight=upside_weight,
        visualize=False
    )
    
    # 3. Retrieve current squad roster
    print("\n📋 Fetching current squad team structure...")
    try:
        my_current_team_ids = get_my_player_ids(manager_id, current_gw)
    except Exception as e:
        print(f"   ⚠️ Could not retrieve live manager squad (Offline Sandbox Fallback active): {e}")
        my_current_team_ids = []
        
    player_details_df = gw_projection_df.drop_duplicates('id_player').set_index('id_player')
    
    if not my_current_team_ids:
        # Generate synthetic/mock squad from top players to ensure operational sandbox planning
        print("   👉 Bootstrapping optimization solver using standard baseline squad...")
        mock_ids = []
        for pos, count in zip(['GKP', 'DEF', 'MID', 'FWD'], [2, 5, 5, 3]):
            pids = players_df[players_df['position'] == pos]['id'].unique()[:count]
            mock_ids.extend([int(x) for x in pids])
        my_current_team_ids = mock_ids

    # 4. Map realizable team values
    current_realizable_value_dict = {}
    my_squad_df = players_df[players_df['id'].isin(my_current_team_ids)]
    for _, row in my_squad_df.iterrows():
        current_realizable_value_dict[int(row['id'])] = float(row['now_cost'])
        
    # Output the current active squad
    print(f"\n✨ ACTIVE IN-PLAY SQUAD ({len(my_current_team_ids)} players):")
    for pid in my_current_team_ids:
        if pid in player_details_df.index:
            p = player_details_df.loc[pid]
            print(f"   - [{p['position']}] {p['web_name']} ({p['team_name']}) - £{p['now_cost']:.1f}M")
        else:
            # Handle standard offline fallback placeholder names
            print(f"   - [Player ID {pid}] £5.0M")
            
    # 5. Set up LP solver parameters
    planning_horizon = 6
    free_transfers = 1
    bank_balance = 3.0
    
    solver_fixed = {'Default': []}
    solver_banned = {'Default': [183, 221, 367, 126]}  # standard bans
    
    print(f"\n⚡ Invoking PuLP Linear Programming Solver (Horizon: {planning_horizon} weeks)...")
    
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
        cvar_weight=0.15,
        return_model=True
    )
    
    # 6. Render optimized strategies
    if solver_results:
        prob, variables = solver_results
        squad = variables['squad']
        starters = variables['starters']
        captain = variables['captain']
        trans_in = variables['trans_in']
        player_ids = variables['player_ids']
        
        gameweeks = list(range(current_gw + 1, np.minimum(current_gw + 1 + planning_horizon, 39)))
        
        print(f"\n🎉 SOLVER OPTIMIZATION SUCCESSFUL!")
        print(f"📈 Overall Squad Utility Index: {pulp.value(prob.objective):.4f}\n")
        
        for t in gameweeks:
            print("=" * 62)
            print(f"📅 GAMEWEEK {t} STRATEGY & LINEUP")
            print("=" * 62)
            
            # Extract selections for gameweek t
            trans_in_ids = [p for p in player_ids if (trans_in[p][t].varValue or 0) > 0.9]
            squad_ids = [p for p in player_ids if squad[p][t].varValue > 0.9]
            starter_ids = [p for p in player_ids if starters[p][t].varValue > 0.9]
            bench_ids = list(set(squad_ids) - set(starter_ids))
            
            # Find the designated captain
            captain_id = None
            try:
                captain_id = max(player_ids, key=lambda p: captain[p][t].varValue or 0)
            except Exception:
                pass
                
            # Render recommended transfers
            if trans_in_ids:
                print("🔄 Suggested Transfers:")
                for idx, in_id in enumerate(trans_in_ids):
                    if in_id in player_details_df.index:
                        in_player = player_details_df.loc[in_id]
                        # Heuristic out mapping
                        out_id = squad_ids[idx] if idx < len(squad_ids) else squad_ids[0]
                        out_player = player_details_df.loc[out_id] if out_id in player_details_df.index else {"web_name": f"ID {out_id}", "position": "POS", "now_cost": 0.0}
                        
                        print(f"   🔴 [OUT] {out_player['web_name']} ({out_player['position']}) £{out_player['now_cost']:.1f}M")
                        print(f"   🟢 [IN]  {in_player['web_name']} ({in_player['position']}) £{in_player['now_cost']:.1f}M")
            else:
                print("🛑 Recommended Transfers: None (Save Free Transfer / Roll)")
                
            # Render Starting XI
            print("\n🏃 Starting XI (Recommended Lineup):")
            xi_df = player_details_df.loc[[x for x in starter_ids if x in player_details_df.index]].copy()
            position_order = ['GKP', 'DEF', 'MID', 'FWD']
            xi_df['position'] = pd.Categorical(xi_df['position'], categories=position_order, ordered=True)
            xi_df = xi_df.sort_values(by=['position', 'custom_score'], ascending=[True, False])
            
            for pid, row in xi_df.iterrows():
                cap_suffix = " 👑 (CAPTAIN)" if pid == captain_id else ""
                print(f"   - {row['position']}: {row['web_name']} ({row['team_name']}) - Proj: {row['custom_score']:.2f}{cap_suffix}")
                
            # Render Bench
            print("\n📋 Bench Roster:")
            bench_df = player_details_df.loc[[x for x in bench_ids if x in player_details_df.index]].copy()
            bench_df['position'] = pd.Categorical(bench_df['position'], categories=position_order, ordered=True)
            bench_df = bench_df.sort_values(by=['position', 'custom_score'], ascending=[True, False])
            
            for pid, row in bench_df.iterrows():
                print(f"   - {row['position']}: {row['web_name']} ({row['team_name']}) - Proj: {row['custom_score']:.2f}")
            print()
    else:
        print("❌ Error: Solver failed to find an optimal team transfer strategy.")

if __name__ == '__main__':
    asyncio.run(main())
