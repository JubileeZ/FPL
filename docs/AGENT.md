# AGENT.md

## Overview
This repository contains a data-driven FPL (Fantasy Premier League) Squad Optimizer. It pulls data from the official FPL API, generates expected points (and ceiling upside) using a walk-forward EMA and multinomial BPS-to-bonus model, and solves for the optimal multi-gameweek transfer plan using Linear Programming (PuLP).

## How to Run the Pipeline
The entire pipeline is run from the `FPL_Dashboard.ipynb` notebook.
1. The notebook mounts Google Drive (if running in Colab) and adds the root directory to `sys.path`.
2. It uses `%autoreload 2` so any changes to the python modules are instantly reflected.
3. It imports logic from the `fpl_engine` python package to fetch data, build features, calculate projections, and run the PuLP solver.

## Module Map
- **FPL_Dashboard.ipynb**: The main entry point for running the optimizer and viewing results/visualizations.
- **fpl_engine/**: The core python package containing the mathematical and solver logic.
  - `data.py`: Functions for fetching from the FPL API and loading cached `.parquet` history.
  - `features.py`: Walk-forward EMA team ratings and Bayesian player adjustments.
  - `scoring.py`: BPS multinomial regressions and `Perf_IDX` generation.
  - `solver.py`: PuLP linear programming logic for the sequential transfer planner.
  - `optimization.py`: Optuna loss functions and duel matrices.
- **FPL_API_QUIRKS.md**: Documents critical inversions in the FPL API (e.g., swapped home/away strengths).
- **FPL_SCORING_RULES.md**: The authoritative document on FPL scoring and what the model can/cannot simulate.

## Don't Touch Zones
- **FPL API Inversion Logic**: Do NOT "fix" the home/away strengths mapping. The API actually swaps them. This logic is correct as implemented and documented in `FPL_API_QUIRKS.md`.
- **Variance Aggregation**: Do NOT change the `ceiling_score` calculation to add standard deviations directly. The current approach ($Var[Total] = \sum Var[Components]$) is mathematically correct.
- **BPS Multinomial Weights**: Do NOT hardcode bonus points. Always use the trained logistic regression model (`_fit_bonus_multinomial`) mapped to expected BPS.

## Maintenance & Logging
- **Decision Log**: You MUST update `DECISION_LOG.md` whenever you:
  - Implement a new architectural pattern.
  - Reject a specific approach after testing.
  - Change a core mathematical formula (e.g., scoring logic, minutes engine).
  - This prevents future sessions from repeating "dead end" research.
