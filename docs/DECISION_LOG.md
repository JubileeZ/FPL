# Decision Log

> [!IMPORTANT]
> **Mandatory for AI Agents**: You MUST update this log whenever a significant design decision is made, a new pattern is adopted, or an approach is rejected after testing. This serves as the "long-term memory" of the project to prevent redundant work in future sessions.

A historical record of approaches we tried, why they succeeded or failed, and established ground rules for the AI.

- **[REJECTED] Hardcoding Bonus Points**: We initially tried recreating the BPS system perfectly. **Why it failed**: The API lacks granular match-level data for passes, crosses, tackles, etc. **Solution**: We use a two-stage calibrated multinomial logistic regression (`_fit_bonus_multinomial`) as a proxy.
- **[REJECTED] Standard Deviation Summing for Ceiling**: We tried adding standard deviations of goals + assists + CS to get the ceiling score. **Why it failed**: It assumed a player peaks in every category simultaneously, creating impossible scores. **Solution**: We calculate Variance independently, sum them ($Var[Total] = \sum Var[Components]$), and then take the square root.
- **[ACCEPTED] EMA for Team Ratings**: Form fluctuates. We implemented a walk-forward EMA for team xG and xGC to balance recency bias with season-long data.
- **[REJECTED] Fixing FPL API Data in the DB**: The `bootstrap-static` endpoint has `strength_attack_home` actually mapping to the away team's true strength. We tried fixing this in data extraction. **Why it failed**: It created endless confusion mapping it back to live matches. **Solution**: Leave the raw data inverted, but explicitly swap it when calculating multipliers (`_away = True Home`, `_home = True Away`).
