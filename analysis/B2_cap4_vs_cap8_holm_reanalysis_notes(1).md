# B2 cap-sensitivity reanalysis

## Data integrity
- Input: `v2_seed_level.csv.gz` from the B2 rho-cap run.
- SHA-256: `6e18fe241aaf03f066bcbe51f742b0241f25be06695b0a59c70233603f2c4612`
- Manifest SHA-256: `6e18fe241aaf03f066bcbe51f742b0241f25be06695b0a59c70233603f2c4612`
- Checksum match: **yes**
- Masking-arm test observations: 8640 seed/pipeline/condition rows.
- Independent test seeds per condition: 96.
- Cap-pairing check: all three cap variants have the same 96 seeds in every condition.
- Cap 8 versus cap 12: identical `J_attack` for every seed in all 30 pipeline × rho × SNR conditions.

## Reanalysis specification
This is a transparent reanalysis, **not a claim to reproduce the missing original B2 Holm calculation**.

- Outcome: `J_attack`.
- Contrast: cap 4 minus cap 8.
- Conditions included in one Holm family: all 30 contrasts (3 pipeline families × 5 rho values × 2 SNR levels).
- Paired test: two-sided Wilcoxon signed-rank, `zero_method='wilcox'`, `method='approx'`.
- Multiplicity: Holm step-down correction over the 30 raw p-values.
- Intervals: paired-seed percentile bootstrap, 20,000 resamples; RNG seed 20261121.
- Sign convention: positive differences favour cap 4; negative differences favour cap 8.

## XS-PH at -6 dB
| rho | Mean cap4-cap8 | Paired-bootstrap 95% CI | W/L/T | Raw Wilcoxon p | Holm p (30) |
|---:|---:|---:|---:|---:|---:|
| 0.4 | +0.009583 | [+0.003750, +0.015781] | 17/1/78 | 0.00327846 | 0.08523989 |\n| 0.5 | -0.068594 | [-0.100833, -0.035313] | 32/62/2 | 0.00015068 | 0.00421913 |\n
## Interpretation
Under this explicitly defined 30-test family, the XS-PH comparison at rho=0.40 and -6 dB is not significant at 0.05, while the comparison at rho=0.50 and -6 dB is significant. These decisions match the manuscript's qualitative interpretation.

The exact Holm values in the current manuscript (0.0756 and 0.0030) were **not reproduced** under this specified Wilcoxon/30-test procedure. The submitted `analyze_supplement_b1_b10.py` produces only descriptive B2 summaries; it does not implement B2 paired testing or B2 Holm correction. Thus the original test definition and multiplicity family are not recoverable from the supplied B2 analysis pipeline. The paper and/or repository should document the original procedure or adopt and document this reanalysis procedure before treating any exact adjusted p-value as final.
