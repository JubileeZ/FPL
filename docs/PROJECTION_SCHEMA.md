# Projection Output Schema

This is the bridge between the ML/Projection engine and the PuLP solver. The solver consumes a DataFrame (`gw_projection_df`) structured as follows.

## Core Identifiers
- `id_player` (int): Unique player ID (from FPL API).
- `gameweek` (int): The gameweek ID (e.g., 1 to 38).
- `position` (str): 'GKP', 'DEF', 'MID', or 'FWD'.
- `team_name` (str): Player's club.
- `now_cost` (float): Current price in £m (e.g., `10.0` = £10.0m).

## Input to Solver
- `Perf_IDX` (float): The expected FPL points for this gameweek (the mean). Used as `points_safe`.
- `ceiling_score` (float): The upside scenario (Mean + 1.5 Std Dev). Often used as `points_cap` (captain target).
- `raw_bonus_multiplier` (float): Value between `[0, 0.15]` based on ownership and ceiling gap, used exclusively by the solver for tie-breaking and transfer cost discounting. Not added to points directly.

## Underlying Model Outputs (Information Only)
- `minutes_IDX` (float): Expected minutes [0, 90].
- `is_coverage_spike` (bool): Flag identifying temporary minutes inflation for backup players.
- `coverage_suppression_applied` (bool): Indicator that a revert-to-baseline discount was active.
- `fixture_attack_multiplier` / `fixture_defence_multiplier` (float): Fixture difficulty scaling.
- `CLEAN_SHEET_INDEX`, `GOAL_INDEX`, `ASSIST_INDEX`, `bonus_component`: The individual point contributions that sum up to `Perf_IDX`.
