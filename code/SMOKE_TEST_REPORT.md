# Smoke-test report (engineering only)

A minimal one-seed execution was completed after adding the BDFHistory adapter.

- SNR: -9 dB.
- Requested rho: 0.2.
- Attack arms: `none` and `masking`.
- Session specification: `fixed:20`.
- Test rounds: 40 (shortened intentionally for smoke testing).
- Test seeds: 1.
- Score-calibration seeds: 2; threshold-calibration seeds: 2.
- Pipelines: NoDefense, Huber, BetaHistory, BDFHistory, OracleRemoval, XS-PH, XS-fixed20, XS-fixed50; XS pipelines were exercised with caps 4 and 8.

Observed engineering checks:

1. The run completed and wrote the expected manifest, seed ledger, threshold/calibration, seed-level, round-level, user-level, identity, and displacement files.
2. The manifest identifies the run as `B1-B10-supplement-instrumented` and documents the BDFHistory adaptation.
3. BDFHistory decision telemetry was emitted; its maximum exclusions in this short smoke test were 8, respecting its configured cap.
4. `run_supplement_b1_b10.py --plan-only` generated all five run commands successfully.
5. The analysis script completed smoke-level B5 telemetry, B6-B9 comparator, Oracle calibration, ROC, and no-attack report generation on the smoke output.
6. All top-level Python source files compiled successfully.

**This was not a scientific evaluation.** The test used one seed, 40 rounds, and tiny calibration pools. It must not be cited as evidence of detection performance, calibration quality, statistical significance, or reviewer-response results.
