# FPL Scoring Rules — 2025/26 Season

> **Purpose**: This document is the authoritative reference for AI agents and models
> working on the FPL Squad Optimizer. It defines every official scoring rule, the
> Bonus Points System (BPS), squad/transfer constraints, chips, and — critically —
> documents which rules we **can** and **cannot** reproduce from available API data,
> along with the modelling assumptions we use as substitutes.

---

## 1. Points Scoring (Official)

### 1.1 Appearance Points

| Action | Points |
|:---|:---:|
| Playing 1–59 minutes | **1** |
| Playing 60+ minutes | **2** |

### 1.2 Attacking Points

| Action | Points |
|:---|:---:|
| Goal scored — GKP / DEF | **6** |
| Goal scored — MID | **5** |
| Goal scored — FWD | **4** |
| Assist (all positions) | **3** |
| Penalty miss | **−2** |

### 1.3 Defensive Points

| Action | Points |
|:---|:---:|
| Clean sheet — GKP / DEF (60+ min only) | **4** |
| Clean sheet — MID (60+ min only) | **1** |
| Clean sheet — FWD | **0** |
| Goals conceded — GKP / DEF (per 2 conceded) | **−1** |

### 1.4 Goalkeeping Points

| Action | Points |
|:---|:---:|
| Every 3 saves | **1** |
| Penalty save | **5** |

### 1.5 Discipline

| Action | Points |
|:---|:---:|
| Yellow card | **−1** |
| Red card | **−3** |
| Own goal | **−2** |

### 1.6 Defensive Contribution (NEW in 2025/26)

A **binary 2-point bonus** awarded when a player meets the defensive action threshold
in a single match. This is **capped at 2 pts per player per match** regardless of how
far above the threshold the player goes.

| Position | Threshold | Actions Counted |
|:---|:---:|:---|
| **DEF** | ≥ 10 | Clearances + Blocks + Interceptions + Tackles |
| **MID / FWD** | ≥ 12 | Clearances + Blocks + Interceptions + Tackles + Ball Recoveries |

> **Note for GKPs**: Goalkeepers are **not eligible** for the defensive contribution bonus
> in the standard points system. The optimizer sets `defcon_thresh = 100` for GKPs
> (effectively unreachable) to model this.

### 1.7 Bonus Points (from BPS)

After every match, the top 3 players by BPS score receive:

| BPS Rank | Bonus Points |
|:---:|:---:|
| 1st | **3** |
| 2nd | **2** |
| 3rd | **1** |

Ties are resolved by sharing: if two players tie for 1st, both receive 3 pts and
the 3rd-place player gets 1 pt. If three players tie for 1st, all receive 3 pts
and no 2nd/3rd bonus is awarded.

### 1.8 Captaincy

- **Captain**: points are **doubled** (×2).
- **Vice-captain**: if the captain does not play, the vice-captain's score is doubled.
- **Triple Captain chip**: captain receives **triple** points (×3) for that GW only.

---

## 2. Bonus Points System (BPS) — Full Criteria

The BPS is an index computed by Opta for every player in every match. It is the
**input** to the bonus point allocation (Section 1.7). Understanding its composition
is critical for modelling `E[bonus]`.

### 2.1 Positive BPS Actions

| Action | BPS |
|:---|:---:|
| Playing 1–60 minutes | **3** |
| Playing 60+ minutes | **6** |
| Goal scored — FWD | **24** |
| Goal scored — MID | **18** |
| Goal scored — GKP / DEF | **12** |
| Scoring a goal from a penalty (all positions) | **12** |
| Assist | **9** |
| Clean sheet — GKP / DEF | **12** |
| Penalty save | **8** |
| Save — shot from inside the box | **3** |
| Save — shot from outside the box | **2** |
| Goal-line clearance | **9** |
| Winning goal | **3** |
| Creating a big chance | **3** |
| Successful tackle | **2** |
| Shot on target | **2** |
| Successful open-play cross | **1** |
| Successful dribble | **1** |
| Key pass | **1** |
| Foul won | **1** |
| Every 2 CBI (clearances + blocks + interceptions) | **1** |
| Every 3 recoveries | **1** |
| Pass completion 70–79% (min 30 passes) | **2** |
| Pass completion 80–89% (min 30 passes) | **4** |
| Pass completion 90%+ (min 30 passes) | **6** |

### 2.2 Negative BPS Actions

| Action | BPS |
|:---|:---:|
| Yellow card | **−3** |
| Red card | **−9** |
| Own goal | **−6** |
| Penalty miss | **−6** |
| Goals conceded — GKP / DEF (per goal) | **−4** |
| Missing a big chance | **−3** |
| Being caught offside | **−1** |
| Shot off target | **−1** |
| Error leading to a goal | **−3** |
| Foul committed | **−1** |

> **Red card note**: The −9 for a red card subsumes any prior yellow card penalty
> (i.e., it is not cumulative with a yellow in the same match).

---

## 3. Squad & Transfer Rules

### 3.1 Squad Composition

| Rule | Value |
|:---|:---:|
| Total squad size | **15** |
| GKP required | **2** |
| DEF required | **5** |
| MID required | **5** |
| FWD required | **3** |
| Max players from one team | **3** |

### 3.2 Starting XI Constraints

| Position | Min Starters | Max Starters |
|:---|:---:|:---:|
| GKP | 1 | 1 |
| DEF | 3 | 5 |
| MID | 2 | 5 |
| FWD | 1 | 3 |

Total starters: always **11**.

### 3.3 Budget

- Starting budget: **£100.0m**
- Player prices fluctuate during the season based on ownership transfer activity.
- **Selling price**: If a player's price has risen since purchase, you receive 50%
  of the profit (rounded down). If it has dropped, you lose the full drop.

### 3.4 Transfers

| Rule | Value |
|:---|:---:|
| Free transfers per GW | **1** (earned each GW) |
| Max banked free transfers | **5** |
| Cost per additional transfer | **−4 pts** |
| GW16 AFCON top-up | Free transfers topped up to **5** |

### 3.5 Chips (2025/26)

Two sets of chips are available: one for GW 1–19, one for GW 20–38.
Chips from the first half **do not** carry over.

| Chip | Effect | Uses per half-season |
|:---|:---|:---:|
| **Wildcard** | Unlimited free transfers for the GW (permanent) | 1 |
| **Free Hit** | Unlimited free transfers for 1 GW only; squad reverts next GW | 1 |
| **Bench Boost** | All 4 bench players' points count | 1 |
| **Triple Captain** | Captain gets ×3 instead of ×2 | 1 |

> **Assistant Manager chip** has been removed for 2025/26.

---

## 4. What Our Model Can & Cannot Reproduce

This section is the **critical bridge** between the official rules above and the
optimizer's actual implementation. Many BPS components are **not available** from the
FPL API or public data, so we use proxy models.

### 4.1 Data We HAVE from the FPL API

The following fields are available per player per match from `element-summary`
(history endpoint) and per player aggregated from `bootstrap-static`:

| Metric | API Field | Used For |
|:---|:---|:---|
| Minutes played | `minutes` | Appearance pts, all rate scaling |
| Goals scored | `goals_scored` | Goal pts, BPS goal component |
| Assists | `assists` | Assist pts, BPS assist component |
| Clean sheets | `clean_sheets` | CS pts, BPS CS component |
| Goals conceded | `goals_conceded` | GC penalty, BPS GC deduction |
| Saves | `saves` | Save pts, BPS save component |
| Bonus points awarded | `bonus` | Target variable for bonus model |
| BPS score (raw) | `bps` | Training target for BPS model |
| Yellow cards | `yellow_cards` | Card penalty pts, BPS card deduction |
| Red cards | `red_cards` | Card penalty pts, BPS card deduction |
| xG (expected goals) | `expected_goals` | Poisson goal model |
| xA (expected assists) | `expected_assists` | Poisson assist model |
| xGC (expected goals conceded) | `expected_goals_conceded` | CS probability, GC penalty |
| Threat (FPL proprietary) | `threat` | Hybrid goal rate |
| Creativity (FPL proprietary) | `creativity` | Hybrid assist rate |
| Influence (FPL proprietary) | `influence` | Legacy metric (not directly used) |
| ICT Index | `ict_index` | Legacy composite (not directly used) |
| Defensive contribution | `defensive_contribution` | DefCon probability model |

### 4.2 Data We DO NOT Have

The following BPS sub-components are **not** exposed by the FPL API at the
individual-match level and therefore **cannot be directly modelled**:

| BPS Component | Why It Matters | Our Workaround |
|:---|:---|:---|
| Pass completion % & attempts | 2/4/6 BPS tiers | **Not modelled** — absorbed into calibration residual |
| Successful tackles | 2 BPS each | **Partially captured** via `defensive_contribution` proxy |
| Successful dribbles | 1 BPS each | **Not modelled** — absorbed into calibration residual |
| Key passes | 1 BPS each | **Partially proxied** via `creativity` metric |
| Big chances created | 3 BPS each | **Partially proxied** via `creativity` metric |
| Shots on target | 2 BPS each | **Partially proxied** via `threat` metric |
| Successful open-play crosses | 1 BPS each | **Not modelled** — absorbed into calibration residual |
| Fouls won | 1 BPS each | **Not modelled** |
| Fouls committed | −1 BPS each | **Not modelled** |
| Goal-line clearances | 9 BPS each | **Not modelled** — very rare event |
| Winning goal bonus | 3 BPS | **Not modelled** — context-dependent |
| Being caught offside | −1 BPS each | **Not modelled** |
| Shots off target | −1 BPS each | **Not modelled** |
| Missed big chances | −3 BPS each | **Not modelled** |
| Errors leading to goal | −3 BPS each | **Not modelled** |
| Penalty saves | 8 BPS, 5 pts | **Not modelled** — too rare |
| Penalty misses | −6 BPS, −2 pts | **Not modelled** — too rare |
| Own goals | −6 BPS, −2 pts | **Not modelled** — too rare |

### 4.3 Our BPS → Bonus Modelling Approach

Since we cannot reconstruct the full BPS score from first principles (too many
hidden sub-components), we use a **two-stage calibrated proxy** approach:

#### Stage 1: Estimate BPS from Available Metrics

We construct an **estimated BPS** using the components we *can* model:

```
estimate_bps =
    minutes_bonus            (3 or 6 BPS based on minutes)
  + goal_bps                 (24/18/12 × xGoals × finishing_factor)
  + assist_bps               (9 × xAssists)
  + cs_bps                   (12 × CS_probability for GKP/DEF)
  + save_bps                 (2.75 × saves_per_90)
  + gc_penalty_bps           (−4 × xGC for GKP/DEF)
  + defcon_bps               (0.5 × defcon_probability)
  + card_bps                 (−3 × YC_rate, −9 × RC_rate)
```

This estimate intentionally **omits** pass completion, dribbles, crosses, fouls,
big chances, etc. — these "hidden" components create a systematic gap between
our estimate and the actual BPS.

#### Stage 2: Per-Position Linear Calibration

To correct for the systematic bias from missing BPS components, we fit a
**per-position linear regression**:

```
actual_bps ≈ scale × estimate_bps + intercept
```

This calibration absorbs the average contribution of hidden BPS components
(e.g., pass completion BPS tends to be higher for MIDs, tackle BPS higher for
DEFs) into position-specific scale and intercept terms.

The calibration is fitted on historical match data where both `estimate_bps` and
actual `bps` are known (players with ≥45 minutes played).

#### Stage 3: Multinomial Logistic Regression for Bonus Points

The calibrated BPS estimate is fed into a **multinomial logistic regression** model:

```
P(bonus = 0, 1, 2, 3 | calibrated_bps)
```

This produces a probability distribution over bonus point outcomes.
Expected bonus = Σ(k × P(bonus = k)) for k ∈ {0, 1, 2, 3}.

**Key insight**: The bonus model is trained on **actual BPS → actual bonus**
relationships, so even though our BPS estimate is imperfect, the calibration
step ensures the mapping is approximately correct in expectation.

### 4.4 Our Defensive Contribution Modelling Approach

The `defensive_contribution` field from the FPL API provides a per-match count
of relevant defensive actions. Our model:

1. Computes a player's **normalized defensive contribution per 90** (adjusted
   for fixture difficulty via `fixture_defence_multiplier`).
2. Models the probability of exceeding the threshold using a **Normal approximation**:
   - Mean = `defcon_per_90 × fixture_defence_mult × (minutes / 90)`
   - Std = `√mean` (Poisson-like variance assumption)
   - `P(defcon ≥ threshold)` = `1 - Φ((threshold - 0.5 - mean) / std)` (continuity correction)
3. Awards `2 × P(defcon ≥ threshold)` expected points.

| Position | Threshold Used |
|:---|:---:|
| GKP | 100 (effectively 0% probability) |
| DEF | 10 |
| MID | 12 |
| FWD | 12 |

### 4.5 Our Clean Sheet Probability Model

Clean sheet probability is modelled using a **Poisson zero-event probability**:

```
P(CS) = exp(−λ)
```

Where `λ = adj_xGC_per_90 × (minutes / 90)`:
- `adj_xGC_per_90` is the player's team-level expected goals conceded rate,
  adjusted for fixture difficulty (`fixture_defence_multiplier`) and the player's
  individual protective factor.
- Only awarded when projected minutes ≥ 60 (FPL rule).

### 4.6 Our Goals & Assists Model

Goals and assists are modelled using **hybrid rates** that blend two signal sources:

```
hybrid_xG_per_90 = w_threat × threat_per_90 + w_xG × xG_per_90
hybrid_xA_per_90 = w_creativity × creativity_per_90 + w_xA × xA_per_90
```

The weights (`w_threat`, `w_xG`, `w_creativity`, `w_xA`) are fitted via
**constrained linear regression** (non-negative coefficients, no intercept) on
historical data, predicting actual goals/assists from xG/threat and xA/creativity.

A **Bayesian finishing factor** adjusts for individual conversion quality:

```
finishing_factor = (goals_scored + C) / (xG + C)
```

Where `C` is a shrinkage constant (default 5) that pulls small-sample players
toward league-average conversion rates.

#### 4.6.1 Cameo Inflation Protection (Bayesian Shrinkage)
To prevent statistical noise from backup players with very few minutes (e.g., a sub recording 0.5 xG in a 1-minute cameo), the engine applies two layers of protection:

1. **Historical Rate Flooring**: When calculating per-match rates for the EMA, minutes are floored at **15 minutes**. This caps the maximum possible rate inflation from a single short appearance.
2. **Season-Long Bayesian Shrinkage**: All per-90 features are calculated with a **90-minute prior** (`per90_shrinkage_mins`):
   ```
   Rate_per_90 = (Total_Metric / (Total_Minutes + 90)) * 90
   ```
   This ensures that a player's individual record is only trusted once they have played significant minutes, pulling outliers toward a conservative zero-baseline.

---

## 5. Summary: Model Coverage vs Official Rules

| Scoring Component | Official Pts | Model Status | Confidence |
|:---|:---:|:---|:---:|
| Appearance (1/2 pts) | 1–2 | ✅ Exact — from minutes projection | High |
| Goals | 4–6 | ✅ Modelled — Poisson on hybrid xG | High |
| Assists | 3 | ✅ Modelled — Poisson on hybrid xA | High |
| Clean sheets | 0–4 | ✅ Modelled — Poisson P(GC=0) | High |
| Goals conceded penalty | −1/2gc | ✅ Modelled — analytical formula | High |
| Saves (GKP) | 1/3s | ✅ Modelled — from saves_per_90 | Medium |
| Yellow cards | −1 | ✅ Modelled — from YC rate | Medium |
| Red cards | −3 | ✅ Modelled — from RC rate | Low (rare) |
| Defensive contribution | 2 | ✅ Modelled — Normal threshold model | Medium |
| Bonus points (1–3) | 1–3 | ⚠️ Proxy — calibrated BPS → multinomial | Medium |
| Penalty saves | 5 | ❌ Not modelled — too rare | N/A |
| Penalty misses | −2 | ❌ Not modelled — too rare | N/A |
| Own goals | −2 | ❌ Not modelled — too rare | N/A |

---

## 7. Stochastic Modeling & Risk (NEW in Optimizer v4)

As of May 2026, the optimizer supports high-performance stochastic pipelines to better model upside and rotation risk.

### 7.1 GARCH Minutes Volatility
Rather than using static trailing standard deviation, the engine uses a **GARCH(1,1)** model to estimate conditional volatility of minutes.
- **Goal**: Capture the "volatility clustering" of rotation-prone players.
- **Usage**: Feeds into the `minutes_IDX` dampening logic. High-volatility players get lower expected scores even with high average minutes.

### 7.2 Covariance-Aware Variance Aggregation
Performance indices are no longer assumed to be independent. The model estimates a pairwise correlation matrix $R$ between scoring components.
- **Aggregation Formula**: $\sigma^2_{total} = S R S^T$ (where $S$ is a diagonal matrix of component standard deviations).
- **Impact**: Improves the accuracy of the `ceiling_score` and dynamic upside metrics by accounting for multi-point event correlations (e.g., Goal ↔ Bonus).

---

## 8. Glossary

| Term | Definition |
|:---|:---|
| **BPS** | Bonus Points System — Opta-powered index that determines bonus point allocation |
| **xG** | Expected Goals — statistical probability of a shot resulting in a goal |
| **xA** | Expected Assists — probability of a pass leading to a goal |
| **xGC** | Expected Goals Conceded — team defensive quality metric |
| **Perf_IDX** | Our model's expected FPL points for a player-fixture combination |
| **ceiling_score** | Perf_IDX + Z × σ (upside scenario using variance aggregation) |
| **custom_score** | Blend of Perf_IDX and ceiling_score used for squad selection |
| **DefCon** | Defensive Contribution — the new 2025/26 2-point bonus |
| **CBI** | Clearances, Blocks, Interceptions — core defensive actions |
| **CBIRT** | CBI + Tackles + (Ball Recoveries for MID/FWD) |
| **EMA** | Exponential Moving Average — used for team ratings and form |
| **Finishing Factor** | Bayesian-shrunk goals/xG ratio measuring conversion quality |
| **Protective Factor** | Bayesian-shrunk GC/xGC ratio for GKP/DEF |

---

*Last updated: 2026-05-15 · Season: 2025/26 · Maintained for AI agent context*
