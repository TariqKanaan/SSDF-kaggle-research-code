# B1-V3 Engineering Validation Report

Date: 2026-11-09

This report records engineering checks of the B1-V3 code package. It is **not a scientific result** and does not provide evidence that XS-PH improves sensing performance.

## Checks completed

1. Python syntax compilation completed for the package's 19 Python source files.
2. `validate_b1v3.py` returned `PASS`:
   - 20 random-balanced session schedules checked;
   - all schedules covered 200 rounds;
   - nominal session-duration count spread was at most one;
   - 20 distinct schedules were generated;
   - development, calibration and test seed pools were checked for overlap;
   - PH CLI parameters, compact B1 telemetry and the closed-block double-logging patch were detected.
3. An end-to-end development orchestration smoke test completed using 3 development seeds, 3 score-calibration seeds, 3 threshold-calibration seeds and 200 rounds. It produced five PH candidate outputs, one shared fixed-K baseline output, and a development selection file. The smoke sample was intentionally small. The observed guardrail failure and sample-size recommendation from this smoke run are not scientific findings and must not be cited.
4. A final-runner and analysis smoke test completed using 2 test seeds and 2 seeds in each calibration pool with the explicit `--allow-underpowered-override` debug flag. The analyzer's integrity check passed. This run is strictly a software-path test; its performance estimates and p-values are not suitable for scientific inference.
5. The Kaggle notebook passed nbformat validation; all code cells parsed successfully with Python AST.

## Important scientific safeguards

- PH parameters are selected using development seeds only.
- A global fixed K is selected from the fixed-only development run.
- Fixed-K baselines are computed once to avoid repeated computation; the selector verifies that candidate PH runs and the fixed baseline run share the same development seeds, session schedules, attacker identities and extreme/mild subgroup assignments.
- Final calibration and test seeds are disjoint from development pools.
- The clean-PFA threshold is calibrated at a nominal 0.05 quantile target; achieved PFA on the held-out test data is measured and reported, not forced to equal 0.05.
- The B1 primary comparison is restricted to the predeclared variable-duration session condition and compares XS-PH with XS-fixed20 and the development-selected fixed K. Fixed-duration conditions are secondary diagnostics.
- Boundary detection metrics are distinct from final sensing-performance metrics.
- The package does not guarantee or assume that the PH performance-improvement hypothesis will be supported.

## Operational note

The full development and final test can be computationally substantial. Do not shrink the final test count to fit a convenient runtime and then describe it as adequately powered. The `--allow-underpowered-override` option is for software debugging only. If the development-derived sample-size plan exceeds practical resources, report the limitation and revise the design transparently before the final test; do not inspect interim final-test outcomes to decide whether more seeds are needed.
