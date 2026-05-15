import json

notebook_path = '/Users/jubilee/Library/CloudStorage/GoogleDrive-z.jubilee.z@gmail.com/My Drive/Hobby/FPL/FPL_Dashboard.ipynb'
with open(notebook_path, 'r') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        for i, line in enumerate(source):
            if 'from fpl_engine.features import compute_rolling_team_ratings, blend_team_ratings, get_fixture_players_stats_df' in line:
                source[i] = line.replace('get_fixture_players_stats_df', 'get_fixture_players_stats_df, compute_global_z_distributions')
            
            if 'blended_team_df = blend_team_ratings' in line:
                source[i] = line.replace('blended_team_df = blend_team_ratings', 'blended_team_df, latest_blended_ratings = blend_team_ratings')
            
            if 'fixture_player_df = get_fixture_players_stats_df' in line:
                source.insert(i, "global_dists = compute_global_z_distributions(blended_team_df)\n")
                source.insert(i+1, "params = {'fixture_alpha_att': 0.15, 'fixture_alpha_def': 0.15, 'recency_ema_alpha': 0.20, 'minutes_ema_alpha': 0.40}\n")
                source[i+2] = line.replace('players_df, blended_team_df', 'params, raw_history_df, global_dists, blended_team_df, latest_blended_ratings')
                break

with open(notebook_path, 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
