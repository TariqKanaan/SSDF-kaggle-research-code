# B10 persistence analysis: standalone Holm re-analysis

## Purpose and scope

This package applies a separate Holm step-down multiplicity correction to the
archived B10 paired-contrast table. It is a standalone secondary re-analysis
of the archived raw Wilcoxon p-values. It does not replace the original B10
simulation, and it is not part of the frozen B1-V3 primary confirmatory test
family.

The script preserves the original contrast columns and appends two fields:
`holm_family_n` and `p_holm_within_metric_family`.

## Exact definition of the Holm test family

**One Holm family is defined separately for each outcome metric.** Each family
contains exactly **64 two-sided paired Wilcoxon tests**, formed by the full
factorial combination:

- **8 pipelines**: BDFHistory, BetaHistory, Huber, NoDefense, TTF-CF, XS-PH,
  XS-fixed20, and XS-fixed50;
- **2 requested malicious-user fractions**: rho = 0.2 and rho = 0.4;
- **2 SNR settings**: -9 dB and -6 dB;
- **2 persistence contrasts**: w = 0.5 versus w = 0, and w = 0.9 versus
  w = 0.

Therefore, the family size is

`8 pipelines × 2 rho values × 2 SNR values × 2 persistence contrasts = 64 tests per metric`.

The four separate metric families are `J_attack`, `auc_attack`, `ffr_clean`,
and `pfa_clean`. Thus the archived table has **4 × 64 = 256 tests in total**,
but Holm correction is **not** applied to one pooled family of 256 tests and
is **not** applied across the four different metrics. The nominal family-wise
significance level is alpha = 0.05.

## Paired design and input fields

Each row is a contrast between one non-zero persistence setting (w = 0.5 or
w = 0.9) and the zero-persistence reference (w = 0), for the same pipeline,
rho, SNR, and seed. Each contrast contains 96 matched test-seed
pairs. The input's `p_wilcoxon_raw` column contains the two-sided paired
Wilcoxon p-value from the original B10 paired analysis.

**What this script recalculates:** Holm-adjusted p-values and the family-size
field only.

**What this script does not recalculate:** the paired seed-level differences,
mean differences, confidence intervals, or raw Wilcoxon p-values. Those
statistics are read from the archived 10-column contrast CSV. Accordingly,
this is an independent multiplicity-correction re-analysis, not a rerun of the
underlying seed-level Wilcoxon tests.

## XS-PH results after metric-wise Holm adjustment

| Metric | Significant XS-PH contrasts (out of 8) |
|---|---:|
| `J_attack` | 4 / 8 |
| `ffr_clean` | 8 / 8 |
| `auc_attack` | 3 / 8 |
| `pfa_clean` | 0 / 8 |

Significance is defined as the Holm-adjusted p-value being below 0.05.
These counts refer to XS-PH's eight combinations (2 rho × 2 SNR × 2
persistence contrasts) within each metric family.

## Method

For each metric separately, raw p-values are sorted in ascending order. For
the p-value at rank `i` (zero-based), the Holm candidate is

`min(1, (m - i) × p_(i))`, where `m = 64`.

The adjusted values are the cumulative maximum of these candidates in sorted
order, then mapped back to the original rows. This is the standard Holm
step-down adjusted-p procedure.

## Reproduction

The script uses only the Python standard library; no package installation is
required. From the repository root, run:

```bash
python B10_Holm_reanalysis_package/reanalyze_B10_holm.py \
  --input analysis/B10_paired_persistence_contrasts.csv \
  --output analysis/B10_paired_persistence_contrasts_Holm_verified.csv \
  --notes analysis/B10_Holm_reanalysis_notes.md
```

The script checks the complete 256-row design, uniqueness of every
metric/pipeline/rho/SNR/persistence contrast, the expected four metric
families of 64 rows each, and the 96-pair count before writing results. It
fails rather than silently proceeding if the input family is incomplete or
contains duplicated contrasts.

## Output fields

- `holm_family_n`: number of tests in the corresponding metric-specific Holm
  family (64 for every row).
- `p_holm_within_metric_family`: Holm-adjusted p-value within that metric's
  64-test family.

The original 10 columns are retained without changing their meaning.
