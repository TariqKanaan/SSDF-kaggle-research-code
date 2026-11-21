# B2 cap-sensitivity reanalysis: reproducible script output

## Input integrity
- Input file: `/mnt/data/v2_seed_level.csv(6).gz`
- Input SHA-256: `6e18fe241aaf03f066bcbe51f742b0241f25be06695b0a59c70233603f2c4612`
- SHA-256 recorded in `v2_output_sha256.json`: `6e18fe241aaf03f066bcbe51f742b0241f25be06695b0a59c70233603f2c4612`
- Checksum match: **yes**
- Rows in complete input: 10,368
- Masking-arm rows in the selected grid: 8,640
- Paired test seeds per contrast: 96
- Cap 8 versus cap 12 mismatched `J_attack` seed values: 0

## Analysis specification
- Outcome: `J_attack`.
- Contrast: cap 4 minus cap 8; positive differences favour cap 4.
- One Holm family of 30 contrasts: 3 pipelines × 5 rho values × 2 SNR levels.
- Paired test: two-sided Wilcoxon signed-rank, `zero_method='wilcox'`, no continuity correction, normal approximation.
- Bootstrap CI: paired-seed percentile bootstrap, 20,000 resamples, NumPy default_rng seed 20261121.
- Bootstrap stream iteration order: SNR, rho, pipeline; this order is fixed in the script.
- Holm: step-down adjustment across all 30 raw p-values.

## Software versions used for this run
- Python: `3.13.5`
- NumPy: `2.3.5`
- pandas: `2.2.3`
- SciPy: `1.17.0`

## XS-PH at -6 dB
| rho | Mean cap4-cap8 | 95% paired-bootstrap CI | W/L/T | Raw Wilcoxon p | Holm p (30) |
|---:|---:|---:|---:|---:|---:|
| 0.10 | +0.000052 | [+0.000000, +0.000156] | 1/0/95 | 0.31731051 | 1 |
| 0.20 | +0.002083 | [+0.000000, +0.005625] | 2/0/94 | 0.17971249 | 1 |
| 0.30 | +0.000781 | [-0.002083, +0.003437] | 4/1/91 | 0.50018426 | 1 |
| 0.40 | +0.009583 | [+0.003750, +0.015781] | 17/1/78 | 0.0032858424 | 0.085431902 |
| 0.50 | -0.068594 | [-0.100833, -0.035313] | 32/61/3 | 0.00015804912 | 0.0044253753 |

## Interpretation and reproducibility note
The inferential family is explicitly defined as the 30 cap-4 versus cap-8 `J_attack` contrasts above. The supplied historical B2 analysis script did not implement these paired B2 tests or their Holm correction; this file provides a standalone, transparent reanalysis from the seed-level data.

The precise asymptotic Wilcoxon p-values can vary slightly across SciPy versions because paired score differences are floating-point values and near-equal absolute differences may be ranked differently. For exact reproduction of this run, use the software versions recorded above. The qualitative decisions for the two highlighted XS-PH cells at rho=0.40 and rho=0.50 remain the same at alpha=0.05.

Confidence intervals are unadjusted paired-bootstrap 95% intervals; Holm adjustment applies to p-values.
