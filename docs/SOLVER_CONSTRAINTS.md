# Solver Constraints

This document defines the rules enforced by the PuLP Linear Programming solver (`FPL_Sequential_Model`). If you modify the solver, you MUST ensure these constraints remain intact to comply with FPL game rules.

## 1. Squad Constraints
- **Total Squad Size**: Exactly 15 players.
- **Position Limits**: Exactly 2 GKP, 5 DEF, 5 MID, 3 FWD.
- **Team Limits**: Maximum 3 players from any single Premier League team.
- **Starting XI**: Exactly 11 starters.
- **Captain**: Exactly 1 captain (must be a starter).
- **Formation Limits**: 
  - GKP: exactly 1 starter
  - DEF: 3 to 5 starters
  - MID: 2 to 5 starters
  - FWD: 1 to 3 starters

## 2. Financial Constraints
- **Budget**: Typically **100.0** (representing £100.0m). The code uses actual £m units.
- **Player Value**: 
  - Purchase cost is `now_cost`.
  - Sale revenue is original purchase price + 50% of profit (rounded down).
  - The model uses a `spread_dict` to calculate the realizable sale value.
- **Bank**: `Bank` variable must be ≥ 0.

## 3. Transfer Constraints
- **Free Transfers**: 
  - Earn 1 Free Transfer (FT) per Gameweek.
  - Can bank up to 5 FTs (introduced in 2025/26 rules).
- **Hits**: Each transfer beyond the available FTs costs exactly -4 points.
- **Wildcard/Free Hit**: Unlimited transfers for 0 points. FT balance resets to 1 after playing.

## 4. Objective Function Structure
The solver maximizes expected utility, potentially blending expected returns with tail-risk protection:
`Maximize: (1 - ω) * [deterministic_utility] + ω * [stochastic_utility]`

### 4.1 Deterministic Utility Components
- **squad_score**: Expected points (Starters + Captain*2 + Bench*0.05).
- **carry_ft_reward**: Reward for banking free transfers (FTs).
- **transfer_costs**: (Paid transfers * 4) + transfer_friction penalty.
- **transfer_hit_discount**: Rebate for "high-value" hits based on bonus model data.
- **bank_penalty**: Minor penalty (e.g. 0.01 per £0.1m) to encourage utilizing funds.
- **starter_diff_tiebreaker**: Micro-weight (0.5 * bonus_mult) to differentiate equal-point options.

### 4.2 Stochastic Utility (CVaR)
If `cvar_weight` (ω) > 0, the solver adds a **Conditional Value at Risk** component:
- **tail_risk**: The average expected points in the worst 10% (p10) of Monte Carlo scenarios.
- **Purpose**: Avoids "glass cannon" squads by penalizing players with extreme downside variance or high rotation risk (captured via GARCH).
