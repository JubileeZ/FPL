# FPL Squad Optimizer

An advanced, mathematics-driven Fantasy Premier League (FPL) squad optimizer and transfer planner. This project uses live data from the official FPL API to generate highly calibrated expected points projections, and leverages Linear Programming to map out multi-gameweek transfer strategies.

## 🚀 Core Features

- **Walk-Forward EMA Team Ratings**: Dynamic team attack and defense strengths computed using exponential moving averages to completely eliminate historical data leakage.
- **Bayesian Player Shrinkage**: Normalizes player stats against fixture difficulty and shrinks small-sample variance (e.g., Goalkeepers and Forwards) toward positional means to prevent overfitting to early-season streaks.
- **Rigorous Variance Math**: Uses mathematically sound variance aggregation ($Var[Total] = \sum Var[Component]$) for Goals, Assists, Clean Sheets, and Bonus Points to accurately model player "Ceiling Scores" (upside) without double-counting joint extremes.
- **Linear Programming Solver (PuLP)**: Plans up to 6 Gameweeks of transfers. Respects FPL constraints including bank balance, Free Transfer banking (up to 5), Wildcards, Free Hits, and transfer hit penalties.
- **S-Curve Differential Pricing**: Automatically discounts the transfer cost of low-ownership players to find optimal differential picks when chasing rank.

## ⚙️ Parameter Tuning & Optimization

The system includes an automated tuning layer (`fpl_tuner.py`) that uses **Optuna** to fine-tune model weights (e.g., fixture difficulty vs. form).

### Initialization from Notebook
To initialize or manually trigger a re-tune from the `FPL_Dashboard.ipynb`:
```python
from fpl_tuner import FPLTuningOrchestrator

# Initialize orchestrator
tuner = FPLTuningOrchestrator(workspace_path="./")

# Run tuning (automated check for 7-day staleness or drift)
tuner.run(force=False)

# Manual override trigger
tuner.run(force=True)
```

### Manual Trigger Function Signatures
- `FPLTuningOrchestrator.run(force: bool = False)`:
  - `force=False` (Default): Only triggers if parameters are > 7 days old or statistical drift is detected.
  - `force=True`: Bypasses all checks and executes a fresh parallelized optimization study.

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable).
2. **Create a Python Virtual Environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install pytest  # For running tests
   ```
   Open `FPL_Dashboard.ipynb` in your preferred environment. Select the `.venv` kernel.

## 📉 Diagnostics & Logging
Long-running optimization tasks are tracked in `logs/`:
- `optimization_metrics.log`: Trial results and parameter shifts.
- `tuning_errors.log`: Failures and edge case reports.
- `state_transitions.log`: History of manual vs. automated updates.

## 🏗️ Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flow and persistence mapping.
