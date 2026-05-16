---
title: Standard Feature Implementation
description: Orchestrates an agent through the linear phase cycle of technical planning, atomic implementation, and subagent browser validation.
---

# Execution Steps

1. **Phase 1: Deep Repository Analysis**
   - Run a discovery scan across workspace files to evaluate existing patterns and dependency definitions.
   - Analyze any ambiguity or architectural edge-cases in the user's requirement brief. 
   - Propose exactly three technical options with clear trade-offs, highlighting one explicit recommendation.

2. **Phase 2: Generate Planning Artifacts**
   - Create an `implementation_plan.md` artifact outlining all files to create, files to modify, external npm dependencies to fetch, and database schema migrations.
   - Pause execution and await user `Proceed` validation before writing to disk.

3. **Phase 3: Sliced Codebase Execution**
   - Enter Fast Mode execution to modify files incrementally.
   - For Git commit generations, invoke the global `@git-commit-formatter` asset path to map structural changes to conventional metrics.

4. **Phase 4: Multi-Environment Verification**
   - Fire up the Antigravity local browser subagent to render modifications.
   - Take structured screenshots at different interactive viewframes to verify UI compliance.
   - Run local terminal test commands (e.g., `npm test` or `vitest`) to ensure zero runtime regressions.

5. **Phase 5: Asynchronous Hand-off**
   - Compile a comprehensive `walkthrough.md` markdown file summarizing absolute progress.
   - Attach browser recordings or diagnostic console summaries directly into the conversation dashboard view.
