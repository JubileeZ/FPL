# ADR 001: Automated Triggering Frequency Policy

## Status
Accepted

## Context
The FPL environment is highly dynamic. Player roles, team tactics, and league-wide scoring trends (the "meta") shift throughout a 38-gameweek season. Static model weights (e.g., the importance of player form vs. fixture difficulty) risk becoming suboptimal as the season progresses.

## Decision
We will implement a dual-trigger policy for re-tuning parameters:
1. **Time-Based Trigger**: Mandatory re-tuning every **7 days** (matching the standard FPL gameweek cadence).
2. **Performance-Based Trigger**: Emergency re-tuning triggered by statistical **Concept Drift** (detected via Page-Hinkley test on scoring residuals).

## Criteria
- **Stability**: Prevent volatile week-to-week swings in solver logic.
- **Compute Cost**: Optimization studies are expensive; they should only run when meaningful data has been added.
- **Adaptability**: The model must capture sudden tactical shifts (e.g., changes in VAR implementation or attacking fullback usage).

## Rationale
- A **7-day threshold** ensures that at least one full set of match data has been processed before the model attempts to find new optimal weights.
- The **Page-Hinkley trigger** allows the system to respond to "shocks" in the data distribution that occur faster than the 7-day cycle.
- **Parameter Smoothing** (weighted averaging of old and new weights) is utilized to ensure that the time-based trigger acts as a low-pass filter, maintaining solver stability over long horizons.

## Consequences
- The system requires a background orchestrator to check staleness before every solve operation.
- Initial solve operations after a 7-day gap will have a latency overhead of ~2-5 minutes as parallel trials execute.
