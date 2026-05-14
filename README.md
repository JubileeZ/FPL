# FPL Squad Optimizer

An advanced, mathematics-driven Fantasy Premier League (FPL) squad optimizer and transfer planner. This project uses live data from the official FPL API to generate highly calibrated expected points projections, and leverages Linear Programming to map out multi-gameweek transfer strategies.

## 🚀 Core Features

- **Walk-Forward EMA Team Ratings**: Dynamic team attack and defense strengths computed using exponential moving averages to completely eliminate historical data leakage.
- **Bayesian Player Shrinkage**: Normalizes player stats against fixture difficulty and shrinks small-sample variance (e.g., Goalkeepers and Forwards) toward positional means to prevent overfitting to early-season streaks.
- **Rigorous Variance Math**: Uses mathematically sound variance aggregation ($Var[Total] = \sum Var[Component]$) for Goals, Assists, Clean Sheets, and Bonus Points to accurately model player "Ceiling Scores" (upside) without double-counting joint extremes.
- **Linear Programming Solver (PuLP)**: Plans up to 6 Gameweeks of transfers. Respects FPL constraints including bank balance, Free Transfer banking (up to 5), Wildcards, Free Hits, and transfer hit penalties.
- **S-Curve Differential Pricing**: Automatically discounts the transfer cost of low-ownership players to find optimal differential picks when chasing rank.

## 📦 Installation & Setup

1. **Clone the repository** (if applicable).
2. **Create a Python Virtual Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Launch the Optimizer**:
   Open `FPL Squad Optimizer Official API V2.ipynb` in your preferred Jupyter environment (VS Code, JupyterLab) and select the `.venv` kernel.

## ⚙️ How It Works

The optimization pipeline flows through several distinct modules:
1. **API Ingestion**: Fetches `bootstrap-static`, `fixtures`, and `element-summary` endpoints.
2. **Feature Engineering**: Normalizes raw stats into per-90 rates and strips away opponent difficulty.
3. **Expected Value Engine**: Calculates `Perf_IDX` (expected points) based on Poisson models for goals/assists/clean sheets and a Multinomial Logistic Regression model for Bonus Points.
4. **Solver Execution**: Passes the projections to the `plan_sequential_transfers` LP function, which outputs the mathematically optimal series of transfers to maximize total expected points over the horizon.

## ⚠️ Known FPL API Quirks
The official FPL `bootstrap-static` API incorrectly swaps home and away team strengths (see `FPL_API_QUIRKS.md` for details). The codebase intentionally intercepts and swaps these values back to maintain mathematical integrity. Do not "fix" this inversion in the code unless FPL officially updates their API.

## 🛠 Tech Stack
- **Data Manipulation**: `pandas`, `numpy`
- **Statistical Modeling**: `scipy`, `scikit-learn`
- **Optimization**: `pulp` (CBC/GLPK solvers)
- **Hyperparameter Tuning**: `optuna`
- **Caching**: `fastparquet` / `pyarrow`
