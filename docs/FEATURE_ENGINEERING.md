# Feature Engineering Spec

This document explains the "why" behind the core columns and metrics in the pipeline to prevent data leaks and double-counting.

## 1. Team-Level Metrics
- **rolling_xG_per90 / rolling_xGC_per90**: Walk-forward EMA of team attack/defence performance. Computed using *past* data only to prevent data leakage into the current fixture.
- **final_attack_rating / final_defence_rating**: Blends the rolling EMA with FPL API strength ratings (fixing the API's inverted home/away quirk).

## 2. Player-Level Adjustment Factors
- **indiv_finishing_factor**: `(goals_scored + C) / (xG + C)`. A Bayesian-shrunk ratio measuring if a player historically out/underperforms their xG.
- **indiv_protective_factor**: `(goals_conceded + C) / (xGC + C)`. (For GKP/DEF only). Measures if a player provides exceptional defensive value beyond the team's baseline.

## 3. Minutes Projection (`minutes_IDX`)
- **Base blend**: Form vs Season average.
- **Bounce-Back Override**: If a proven starter was fully fit but played 0 mins, they were likely rested. Projects a bounce-back.
- **Backup Anomaly Filter**: Detects if recent form is a temporary spike for a depth rank 2+ player (stand-in). Reverts projected minutes toward the historical baseline using a tunable convex blend.
- **Injury Discount**: Scales down minutes probabilistically based on chance_of_playing_next_round.
- **Why it matters**: minutes_IDX is the fundamental scaling denominator. All expected points (goals, assists, CS) are scaled by (minutes_IDX / 90).

## 4. Expected Points (`Perf_IDX`)
- **CLEAN_SHEET_INDEX**: Uses Poisson probability `exp(-adj_xGC_pred)`.
- **CONCEDED_PENALTY**: Exact analytical formula for the discrete `-1pt per 2 goals conceded` FPL rule.
- **defcon_component**: Uses a normal approximation to find the probability of a player exceeding the 10 or 12 defensive actions threshold.
