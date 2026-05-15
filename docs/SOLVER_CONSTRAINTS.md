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
The solver maximizes expected utility:
`Maximize: squad_score + carry_ft_reward - transfer_costs + transfer_hit_discount - bank_penalty + starter_diff_tiebreaker`
- **squad_score**: Expected points (Starters + Captain*2 + Bench*0.05).
- **transfer_costs**: (Paid transfers * 4) + transfer_friction penalty.
- **bank_penalty**: Minor penalty (e.g. 0.01 per £0.1m) to encourage utilizing funds.
- **starter_diff_tiebreaker**: Micro-weight (0.5 * bonus_mult) applied to starters to differentiate equal-point options.
