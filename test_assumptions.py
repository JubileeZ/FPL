import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('assets/raw_history_cache.parquet')
    
    # Filter for meaningful minutes to analyze distributions
    df_active = df[df['minutes'] >= 45].copy()
    
    # Identify GKPs by checking who has ever made a save
    gkp_ids = df.groupby('id_player')['saves'].max()
    gkp_ids = gkp_ids[gkp_ids > 0].index
    df_gkp = df_active[df_active['id_player'].isin(gkp_ids)]
    
    components = [
        ('Goals', 'goals_scored', df_active),
        ('Goals Conceded', 'goals_conceded', df_active),
        ('Saves (GKPs only)', 'saves', df_gkp),
        ('Defensive Contribution', 'defensive_contribution', df_active),
    ]
    
    print("=== Distribution Assumption Tests ===")
    for name, col, data in components:
        if col not in data.columns:
            print(f"{name}: Column '{col}' not found.")
            continue
            
        series = data[col].dropna()
        if len(series) == 0:
            print(f"{name}: No data.")
            continue
            
        mean = series.mean()
        var = series.var()
        dispersion = var / mean if mean > 0 else 0
        
        print(f"\n--- {name} ({col}) ---")
        print(f"Mean (λ): {mean:.4f}")
        print(f"Variance: {var:.4f}")
        print(f"Dispersion Ratio (Var/Mean): {dispersion:.4f}")
        
        if dispersion > 1.3:
            print("Verdict: OVERDISPERSED (Consider Negative Binomial)")
        elif dispersion < 0.7:
            print("Verdict: UNDERDISPERSED (Consider Binomial)")
        else:
            print("Verdict: POISSON ASSUMPTION HOLDS (Ratio ≈ 1)")
            
except Exception as e:
    print(f"Error: {e}")
