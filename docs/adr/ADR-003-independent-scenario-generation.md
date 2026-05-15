# ADR 003: Independent Player-Specific Scenario Generation

## Status
Accepted

## Context
Scenario generation was previously implemented using a "broadcast" pattern: a single correlated random vector was generated for a gameweek and applied identically to every player in the dataset. This mathematically implied that all player performances were 100% correlated—if the model "rolled" a goal for one player, it essentially increased the likelihood of a goal for every other player in that scenario.

## Decision
We have refactored `generate_scenario_tensor` to use **Independent Stochastic Draws per Player**:
1.  **Internal Correlation**: We maintain the joint correlation between scoring components (Goals, Assists, Bonus, etc.) for *each* individual player using a Cholesky decomposition of the global component correlation matrix.
2.  **Squad Independence**: Each player receives their own unique set of random draws, ensuring that their performance is decoupled from their teammates and opponents in the simulation.

## Rationale
-   **Risk Diversification**: Accurately modeling a diversified squad requires that player outcomes be independent. 100% correlation renders Conditional Value at Risk (CVaR) and standard deviation calculations for the squad useless, as it overestimates the "tail risk" of the entire squad failing simultaneously.
-   **Solver Realism**: By moving to independent draws, the solver can now value "hedging" (e.g., choosing a reliable GKP to offset a volatile FWD), which was impossible under the broadcasted model.

## Consequences
-   The scenario tensor shape is now strictly $(Players, Gameweeks, Scenarios)$.
-   Correlation between any two players in a given gameweek/scenario now averages ~0.0 (as verified in the Phase 3 test suite).
-   CVaR calculations for squad selection are now statistically valid and actionable for risk-averse managers.
