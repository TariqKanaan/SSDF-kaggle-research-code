# Code and results: adaptive temporal segmentation in cooperative spectrum sensing

This repository accompanies the manuscript *An Empirical Performance Evaluation of Adaptive Temporal Segmentation in Cooperative Spectrum Sensing*. It provides the simulation code, analysis scripts, documented seed ranges, experiment outputs, and integrity records that are currently published in the repository.

The GitHub repository is the actively maintained source. A repository version is archived on Zenodo (DOI: [10.5281/zenodo.24114051](https://doi.org/10.5281/zenodo.24114051)); the Zenodo record should be treated as the archived snapshot associated with that DOI, not automatically as a mirror of every later GitHub commit.

## Repository structure

| Path | Contents | Manuscript location |
|---|---|---|
| `code/` | Simulator, XS-PH and comparator implementations, experiment runners, analysis and validation scripts, notebooks, and package requirements | All experiments |
| `B1V3/` | Primary-experiment performance and boundary summaries, paired-test results, frozen development-selection record, session-integrity results, and integrity report | Sections 3.1–3.2; Tables 2–4; Appendix E |
| `B2_rho_cap/` | Cap/fraction grid outputs, including seed-level results, seed ledger, manifest, and output SHA-256 index | Section 3.3; Table 5; Appendix F |
| `B2_extreme_mild_split/` | Extreme/mild attacker-composition experiment outputs | Section 3.3; Table 6; Fig. 2 |
| `B5_telemetry_replay/` | Telemetry replay outputs and summaries for the 48-seed mechanism analysis | Section 3.4; Table 7 |
| `B6_B7_B8_B9_comparators/` | Comparator-study outputs and OracleRemoval calibration records | Sections 3.5–3.6; Tables 8–10; Appendices C–D |
| `B10_persistence/` | Persistent-nuisance stress-test outputs | Section 3.7; Table 11; Fig. 3 |
| `analysis/` | Derived summary tables, paired contrasts, and generated analysis reports | Supplemental analyses and tables |
| `B2_cap4_vs_cap8_reanalysis_package/analysis/` | Standalone B2 cap-sensitivity reanalysis script, pinned Python package versions, and analysis audit note | Section 3.3; Table 5; Appendix F |
| `supplementary/` | `Supplementary_Table_S1.pdf` and `Supplementary_Table_S1.csv` | Appendix A; Supplementary Table S1 |
| `validation/` | Supplemental-run integrity report | Reproducibility checks |

The primary `B1V3/` directory currently contains the primary summary outputs, paired-test table, frozen selection record, and integrity checks. Supplemental run directories contain their associated run-level outputs and audit files. Their `v2_manifest.json` files describe the run configuration and source files, while `v2_output_sha256.json` records checksums for the expected output files. The SHA-256 index should be used to verify that every listed file is present and unchanged.

The raw user-level telemetry tables from the B5 replay (approximately 5.4 million rows) are not included because of their size. The associated summaries, replay code, and seed information are provided so that the analysis can be regenerated.

## Requirements

The main experiment code uses Python 3.10 or later and the packages listed in `code/requirements.txt`:

- `numpy>=1.24`
- `pandas>=2.0`
- `scipy>=1.10`
- `scikit-learn>=1.3`

The simulation suite was executed on Kaggle. The complete suite took approximately 3.5 hours in the reported environment.

The separate B2 cap-sensitivity reanalysis has a pinned environment in `B2_cap4_vs_cap8_reanalysis_package/analysis/requirements_b2_reanalysis.txt` to make the Wilcoxon p-values reproducible across software versions.

## Reproducing the primary experiment

Run the following commands from the repository root:

```bash
cd code
python validate_b1v3.py
python run_b1v3_development.py --out ../dev_out
python select_b1v3_development.py --dev-root ../dev_out
python run_b1v3_final.py --selection ../dev_out/B1V3_selection.json --out ../final_out
python analyze_b1v3.py --results ../final_out --selection ../dev_out/B1V3_selection.json
```

The primary session schedule is `random_balanced:13,20,30,50`, with 96 paired test seeds per SNR, 200 rounds per trajectory, and the 0.95 empirical quantile as the decision threshold. The final run also generates fixed-duration schedules (13, 20, 30, and 50 rounds), the fixed-K grid (K = 10, 13, 20, 30, 50), and the `fixed:200` control used for secondary analyses.

## Reproducing the supplemental experiments

Run from the repository root:

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

The supplemental experiments use fixed 20-round sessions, 150 score-calibration seeds, 150 threshold-calibration seeds, and 96 test seeds per condition; the B5 telemetry replay uses 48 test seeds.

## Reproducing the B2 cap-sensitivity reanalysis

The B2 cap-4 versus cap-8 analysis is a separately documented reanalysis of `B2_rho_cap/v2_seed_level.csv.gz`. It defines one family of 30 paired `J_attack` contrasts (3 pipelines × 5 values of ρ × 2 SNR levels), uses two-sided Wilcoxon signed-rank tests with a normal approximation and no continuity correction, and applies Holm adjustment across those 30 tests. The script verifies the input SHA-256, the 96-seed pairing, and the cap-8/cap-12 `J_attack` identity before writing the output files.

Run from the repository root:

```bash
python -m pip install -r B2_cap4_vs_cap8_reanalysis_package/analysis/requirements_b2_reanalysis.txt
python B2_cap4_vs_cap8_reanalysis_package/analysis/recompute_B2_cap4_vs_cap8_holm.py
```

The script writes the 30-contrast CSV and its Markdown audit note to the top-level `analysis/` directory. The pinned environment used for the reported output is Python 3.13.5, NumPy 2.3.5, pandas 2.2.3, and SciPy 1.17.0. Do not manually edit the generated p-values; rerun the script to regenerate them. The confidence intervals are unadjusted paired-bootstrap 95% intervals; Holm adjustment applies to the p-values.

This is a transparent independent reanalysis. It is not represented as the original historical B2 testing procedure: the original supplemental analysis script produces descriptive B2 summaries and does not contain this B2 paired-test/Holm calculation.

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

The experiment seed pools are disjoint. The B5 telemetry replay reuses a subset of the B2 extreme/mild split seeds and must not be counted as additional independent test seeds.

## Notes for readers

- **Terminology in the code.** Some legacy files (`xs_config.py`, `supplement_protocol.json`, and `plan_seed_count.py`) use the working labels “pre-registered” and “confirmatory”. These are legacy labels. The study was not prospectively registered; the frozen specification and hash checks establish code/data consistency only. The manuscript describes the evaluation accordingly.
- **ParadigmA.** `paradigm_a_final.py` and the `ParadigmA` entry in the default pipeline list belong to an earlier version of the study. ParadigmA is not evaluated in the revised manuscript.
- **Study-adapted comparators.** BetaHistory and BDFHistory are transparent adaptations of reputation mechanisms to continuous reports; they are not reproductions of the complete published procedures. TTF-CF is the authors’ earlier, unpublished pipeline and is specified in Appendix B.
- **Telemetry.** The raw user-level B5 observation tables are not included because of their size. The summaries under `analysis/` are included, and the replay can be regenerated using the supplied code and seeds.
- **Planning aids.** `plan_seed_count.py` and `v2_seed_count_plan_from_v1.csv` belong to an earlier planning step. The final primary sample size follows `select_b1v3_development.py`.
- **Integrity.** Integrity reports document engineering checks; they are not scientific results. Use the relevant run's SHA-256 index to verify its published output files.

## Licence

The code is released under the MIT Licence (see `LICENSE`). Result files are released under CC BY 4.0.

## Citation

Please cite the manuscript and the Zenodo record (DOI: 10.5281/zenodo.24114051).
