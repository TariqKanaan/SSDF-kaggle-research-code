# Code and results: adaptive temporal segmentation in cooperative spectrum sensing

This repository accompanies the manuscript *An Empirical Performance Evaluation of Adaptive Temporal Segmentation in Cooperative Spectrum Sensing*. It contains the simulator, the analysis scripts, the seed ledgers and the result files of every experiment reported in the manuscript. An archived copy is available on Zenodo (DOI: 10.5281/zenodo.24114051).

## Contents

| Path | Content | Manuscript |
|---|---|---|
| `code/` | Simulator, XS-PH and comparator pipelines, runners, analysis and validation scripts (unpacked from `XS_PH_B1_B10_SUPPLEMENT_PACKAGE.zip`) | all |
| `B1V3_primary/` | Frozen primary experiment: development selection, final run outputs, paired tests | Sections 3.1–3.2, Tables 2–4, Appendix E |
| `B2_rho_cap/` | Exclusion-cap and attacker-fraction grid (ρ = 0.1–0.5, caps 4/8/12) | Section 3.3, Table 5, Appendix F |
| `B2_extreme_mild_split/` | Extreme/mild composition at ρ = 0.2 and 0.4 | Section 3.3, Table 6, Fig. 2 |
| `B5_telemetry_replay/` | Replay of 48 seeds with user-level telemetry (summaries only) | Section 3.4, Table 7 |
| `B6_B7_B8_B9_comparators/` | 17 pipelines, OracleRemoval calibration, clean state, ROC | Sections 3.5–3.6, Tables 8–10, Appendices C–D |
| `B10_persistence/` | Persistent-nuisance stress test (w = 0, 0.5, 0.9) | Section 3.7, Table 11, Fig. 3 |
| `analysis/` | Summary tables and paired contrasts computed from the runs above | all supplemental tables |
| `validation/` | Integrity report of the supplemental runs | — |

Each run folder holds gzipped CSV files (seed-level results, thresholds, calibration scores, score-bank sizes, session and block boundaries, seed ledgers), a manifest and a SHA-256 file (`v2_output_sha256.json`).

## Requirements

Python 3.10 or later with `numpy>=1.24`, `pandas>=2.0`, `scipy>=1.10` and `scikit-learn>=1.3` (see `code/requirements.txt`). The runs were executed on Kaggle; the complete suite took about 3.5 hours.

## Reproducing the primary experiment

```bash
cd code
python validate_b1v3.py
python run_b1v3_development.py --out ../dev_out
python select_b1v3_development.py --dev-root ../dev_out
python run_b1v3_final.py --selection ../dev_out/B1V3_selection.json --out ../final_out
python analyze_b1v3.py --results ../final_out --selection ../dev_out/B1V3_selection.json
```

The primary session schedule is `random_balanced:13,20,30,50`, with 96 paired test seeds per SNR, 200 rounds per trajectory and the 0.95 empirical quantile as decision threshold. The final run also produces the fixed-duration schedules (13, 20, 30 and 50 rounds), the fixed-K grid (K = 10, 13, 20, 30, 50) and the fixed:200 control used for the secondary analyses.

## Reproducing the supplemental experiments

```bash
cd code
python run_supplement_b1_b10.py --plan-only
python run_supplement_b1_b10.py --run B2_rho_cap
python run_supplement_b1_b10.py --run B2_extreme_mild_split
python run_supplement_b1_b10.py --run B5_telemetry_replay
python run_supplement_b1_b10.py --run B6_B7_B8_B9_comparators
python run_supplement_b1_b10.py --run B10_persistence
python analyze_supplement_b1_b10.py --root <output root>
python validate_supplement.py --root <output root>
```

All supplemental runs use fixed 20-round sessions, 150 score-calibration seeds, 150 threshold-calibration seeds and 96 test seeds per condition (48 for the telemetry replay).

## Seed pools

| Purpose | Score calibration | Threshold calibration | Test |
|---|---|---|---|
| Primary (final) | 72000–72149 | 72200–72349 | 73000–73095 |
| Primary (development) | 65000– | 65100– | 64000–64039 |
| B2 attacker-fraction and cap grid | 84000–84149 | 84200–84349 | 85000–85095 |
| B2 extreme/mild split | 84400–84549 | 84600–84749 | 85100–85195 |
| B5 telemetry replay | 84400–84549 | 84600–84749 | 85100–85147 (subset of the split) |
| B6–B9 comparators | 86000–86149 | 86200–86349 | 87000–87095 |
| B10 persistence | 88000–88149 | 88200–88349 | 89000–89095 |

All pools are disjoint. The telemetry replay repeats a subset of the split experiment and must not be counted as additional independent seeds.

## Notes for readers

* **Terminology in the code.** Some files (`xs_config.py`, `supplement_protocol.json`, `plan_seed_count.py`) use the working labels “pre-registered” and “confirmatory”. These are legacy labels. The study was not prospectively registered, and the frozen specification and hash checks establish code and data consistency only. The manuscript describes the evaluation accordingly.
* **ParadigmA.** `paradigm_a_final.py` and the `ParadigmA` entry in the default pipeline list belong to an earlier version of the study. ParadigmA is not evaluated in the manuscript.
* **Study-adapted comparators.** BetaHistory and BDFHistory are transparent adaptations of reputation mechanisms to continuous reports. They are not reproductions of the complete published procedures. TTF-CF is the authors’ earlier, unpublished pipeline and is specified in Appendix B of the manuscript.
* **Telemetry.** The raw user-level observation tables of the B5 replay (about 5.4 million rows) are not included because of their size. The summaries in `analysis/` (`B5_*`) are included, and the replay can be regenerated with the code and seeds above.
* **Planning aids.** `plan_seed_count.py` and `v2_seed_count_plan_from_v1.csv` come from an earlier planning step. The sample size of the final experiment follows `select_b1v3_development.py`.
* **Integrity.** `b1v3_validation_report.json` and `validation/SUPPLEMENT_INTEGRITY_REPORT.json` record engineering checks. They are not scientific results.

## Licence

The code is released under the MIT licence (see `LICENSE`). Result files are released under CC BY 4.0.

## Citation

Please cite the manuscript and the Zenodo record (DOI: 10.5281/zenodo.24114051).
