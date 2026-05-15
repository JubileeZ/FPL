# FPL API Known Quirks & Issues

## 1. The Naming Inversion of Team Strengths
**Status:** Confirmed (Active as of current FPL API version).

### Description
In the official FPL `bootstrap-static` API, the fields representing a team's home and away strengths are inverted. The FPL developers accidentally mapped the "home" strength to the "away" variable, and vice versa. 

Because football teams are universally stronger when playing at home, their true home strength should be a higher numerical value than their true away strength. However, the API returns the opposite.

### Evidence
When querying top teams (like Arsenal, Man City, Liverpool) from `https://fantasy.premierleague.com/api/bootstrap-static/`, the API returns:

```json
"Arsenal": {
  "strength_attack_home": 1340,
  "strength_attack_away": 1390,
  "strength_defence_home": 1270,
  "strength_defence_away": 1320
}
```

As seen above, the "away" numbers are systematically higher than the "home" numbers. This is logically inverted for real-world football.

### Required Handling in Code
Whenever an AI Agent or developer is mapping these values to calculate fixture difficulties or team ratings, they **MUST** swap them back to reflect reality:

```python
# CORRECT MAPPING TO FIX API INVERSION:
true_home_attack_strength = team_data['strength_attack_away']
true_away_attack_strength = team_data['strength_attack_home']

true_home_defence_strength = team_data['strength_defence_away']
true_away_defence_strength = team_data['strength_defence_home']
```

Failure to apply this inversion fix will result in the model assuming teams play better on the road, which will ruin expected points models and clean sheet probabilities.

*Note for AI Agents: The FPL Squad Optimizer codebase already handles this inversion correctly in `blend_team_ratings` and `get_fixture_players_stats_df`. Do not "fix" it by removing the inversion logic, as it is mathematically necessary to counter the API's bug.*
