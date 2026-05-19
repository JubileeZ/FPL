# FPL Squad Optimizer

An advanced, mathematics-driven Fantasy Premier League (FPL) squad optimizer and transfer planner. This project uses live data from the official FPL API to generate highly calibrated expected points projections, and leverages Linear Programming to map out multi-gameweek transfer strategies.

## 🚀 Core Features

- **Walk-Forward EMA Team Ratings**: Dynamic team attack and defense strengths computed using exponential moving averages to completely eliminate historical data leakage.
- **Bayesian Player Shrinkage**: Normalizes player stats against fixture difficulty and shrinks small-sample variance toward positional means to prevent overfitting to early-season streaks.
- **Hybrid Skellam-Normal Duels**: Evaluates player-vs-player performance using discrete Skellam distributions with an automated Normal-fallback for numerical stability in lopsided matchups.
- **Independent Stochastic Scenarios**: Generates 5,000+ parallel timelines with player-independent random draws, enabling accurate Conditional Value at Risk (CVaR) calculations for squad diversification.
- **Backup Anomaly Detection**: Generalized statistical filter that flags temporary minutes inflation for depth players covering injuries, reverting projections toward historical baselines.
- **Linear Programming Solver (PuLP)**: Plans up to 6 Gameweeks of transfers. Respects FPL constraints including bank balance, Free Transfer banking (up to 5), Wildcards, Free Hits, and transfer hit penalties.

## ⚙️ Parameter Tuning & Optimization

The system includes an integrated tuning engine (`fpl_engine/tuning.py`) that uses **Optuna** to fine-tune model weights (e.g., fixture difficulty vs. form).

### Usage from Notebook
The system automatically manages its own parameters, but you can manually trigger a re-tune:
```python
from fpl_engine.tuning import auto_tune_if_needed

# Bypasses staleness checks and executes a fresh optimization study
await auto_tune_if_needed(current_gw=35, force=True)
```

### Staleness Policy
- **Time-Based**: Parameters are re-tuned if the last update was > 7 days ago.
- **Drift-Based**: Detects statistical shifts in scoring residuals to trigger emergency recalibration.

## 🛠️ Installation & Setup

1. **Clone the repository**.
2. **Create a Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   *(Note: On macOS, use `python3` to initialize the environment. Once activated, the standard `python` command will map directly to this virtual environment).*
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```
   Or launch it directly using the virtual environment executables:
   ```bash
   ./.venv/bin/streamlit run app.py
   ```
5. **Run Validation Suite**:
   ```bash
   python tests/test_statistical_engine.py
   ```

## 📉 Diagnostics & Logging
- `scratch/test_logs.txt`: Verification logs for statistical engine stability.
- `tuned_params.json`: The live configuration file for all model weights.

## ⚠️ Troubleshooting & Environment Stability

### Python 3.14+ pre-release Compatibility & String Corruption
If the Streamlit dashboard starts with random or corrupted text (dynamic import issues, memory faults from scientific C-extension binary wheels like `numpy`, `pandas`, `pyarrow`, `pulp`), it is likely due to stale compiled python bytecode generated across different Python interpreter versions.

To resolve bytecode-induced rendering or execution failures:
1. **Purge Compiler Caches**: Run a recursive command to completely sweep out all stale python caching directories and `.pyc` compiled files:
   ```bash
   find . -type d -name "__pycache__" -exec rm -rf {} +
   find . -name "*.pyc" -delete
   ```
2. **Standardize Virtual Environments**: Ensure you run the application in standard, production-ready environments (e.g. Python 3.12 or 3.13) to avoid C-extension dynamic loading bugs on experimental runtimes.

### High-Fidelity Sandbox & Offline Fallback Mode
When running in restricted sandbox setups (where port/socket binding is blocked or local networks are offline), the optimizer is retrofitted with:
- **Synthetic Data Generation**: Automatically falls back to synthetic squad data frames if the FPL API is unreachable.
- **Local Parquet Fallback**: Loads historical fixture models from `assets/raw_history_cache.parquet`.
- **Defensive tuning guards**: Hyperparameter optimization failsafe logic prevents any network timeouts during Optuna runs from crashing the Streamlit process, gracefully falling back to preserving last stable values in `tuned_params.json`.

## 🏗️ Architecture
See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed data flow and mathematical ADRs in `docs/adr/`.

