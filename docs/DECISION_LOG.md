# Decision Log

> [!IMPORTANT]
> **Mandatory for AI Agents**: You MUST update this log whenever a significant design decision is made, a new pattern is adopted, or an approach is rejected after testing. This serves as the "long-term memory" of the project to prevent redundant work in future sessions.

A historical record of approaches we tried, why they succeeded or failed, and established ground rules for the AI.

- **[REJECTED] Hardcoding Bonus Points**: We initially tried recreating the BPS system perfectly. **Why it failed**: The API lacks granular match-level data for passes, crosses, tackles, etc. **Solution**: We use a two-stage calibrated multinomial logistic regression (`_fit_bonus_multinomial`) as a proxy.
- **[REJECTED] Standard Deviation Summing for Ceiling**: We tried adding standard deviations of goals + assists + CS to get the ceiling score. **Why it failed**: It assumed a player peaks in every category simultaneously, creating impossible scores. **Solution**: We calculate Variance independently, sum them ($Var[Total] = \sum Var[Components]$), and then take the square root.
- **[ACCEPTED] EMA for Team Ratings**: Form fluctuates. We implemented a walk-forward EMA for team xG and xGC to balance recency bias with season-long data.
- **[REJECTED] Fixing FPL API Data in the DB**: The `bootstrap-static` endpoint has `strength_attack_home` actually mapping to the away team's true strength. We tried fixing this in data extraction. **Why it failed**: It created endless confusion mapping it back to live matches. **Solution**: Leave the raw data inverted, but explicitly swap it when calculating multipliers (`_away = True Home`, `_home = True Away`).
- **[ACCEPTED] Hybrid Skellam-Normal Duels**: Numerical stability in duel evaluations. **Decision**: Use Skellam for exact discrete probabilities in balanced matchups, but fall back to a Continuous Normal approximation ($N(\mu_{diff}, Var_{diff})$) in lopsided matchups (high mean-diff/low variance) to prevent degenerate parameters.
- **[REJECTED] Broadcasted Scenarios**: We previously applied the same random seed to all players in a scenario. **Why it failed**: Mathematically implied 100% correlation between all players, making risk metrics (CVaR) invalid. **Solution**: Switched to independent stochastic draws per player while preserving internal component correlation via Cholesky decomposition.

## Scoring Component Distribution Assumptions

**Context:** 
We historically assumed Poisson distributions (Variance = Mean) for predicting rare events (Goals, Assists, Saves, Defensive Contributions). We wanted to empirically test these assumptions against historical data to ensure accurate right-tail modelling (which impacts upside and ceiling scores).

**Decision:**
Implemented a dynamic distribution assumption tester `_check_distribution_assumptions()` that measures the dispersion ratio (Var/Mean) during the execution pipeline. Based on historical data runs:
- **Goals and Assists:** Dispersion ratio is ~1.1, meaning the Poisson assumption holds reasonably well.
- **Goals Conceded:** Dispersion ratio is ~0.97, meaning the Poisson assumption holds.
- **Defensive Contribution (DefCon):** Dispersion ratio is highly overdispersed at ~2.9x (Variance is almost 3x the Mean).

**Logic Adjustment:**
Because DefCon is overdispersed (behaves closer to Negative Binomial than Poisson), using exact Poisson survival functions vastly underestimates the probability of large hauls. The scoring engine was adjusted back to use a Normal approximation (`stats.norm.sf`), but the standard deviation is now dynamically scaled by the measured dispersion factor (`np.sqrt(mean * defcon_dispersion)`). This correctly thickens the right tail to reflect the empirical probability of a player exceeding the BPS defensive threshold.
