# Project Verification

## Goal
Verify that the statistical engine, solver, and pipeline are operational and producing valid FPL strategies.

## Tasks
- [ ] Task 1: Run core statistical tests → Verify: `python tests/test_statistical_engine.py` prints [PASS] for all tests.
- [ ] Task 2: Run solver regression tests → Verify: `python tests/test_solver_regression.py` prints "checks passed."
- [ ] Task 3: Run pipeline CLI → Verify: `python run_pipeline_cli.py` generates a 6-week strategy without errors.
- [ ] Task 4: Verify test logs → Verify: `scratch/test_logs.txt` exists and contains pass status.

## Done When
- [ ] All standalone test scripts execute successfully.
- [ ] CLI runner produces a coherent transfer plan.
