# B10 standalone Holm re-analysis package

This folder contains a reproducible, standalone multiplicity-correction re-analysis of the archived B10 paired-contrast table.

## Files

- `reanalyze_B10_holm.py` — independent Holm step-down implementation; uses only the Python standard library.
- `B10_paired_persistence_contrasts.csv` — original 10-column B10 paired-contrast input (256 rows; raw Wilcoxon p-values and the previously computed paired summaries).
- `B10_paired_persistence_contrasts_Holm_verified.csv` — 12-column output, retaining the original 10 columns and appending `holm_family_n` and `p_holm_within_metric_family`.
- `B10_Holm_reanalysis_notes.md` — full definition of the four metric-specific Holm families, method, scope, and XS-PH significance counts.
- `SHA256SUMS.txt` — file checksums for this package.

## Run

From this folder:

```bash
python reanalyze_B10_holm.py
```

Or specify explicit input/output paths:

```bash
python reanalyze_B10_holm.py \
  --input B10_paired_persistence_contrasts.csv \
  --output B10_paired_persistence_contrasts_Holm_verified.csv \
  --notes B10_Holm_reanalysis_notes.md
```

No `pip install` step is needed.

## Test-family definition

Holm adjustment is performed **separately for each metric**, with 64 tests per metric:

`8 pipelines × 2 rho values × 2 SNR levels × 2 persistence contrasts = 64`.

The four metrics are `J_attack`, `auc_attack`, `ffr_clean`, and `pfa_clean`. Therefore, the table contains 256 tests in four independent Holm families; it is **not** one pooled family of 256 tests. See `B10_Holm_reanalysis_notes.md` for the full design and limitations.

## Important scope limitation

The script independently recalculates the Holm-adjusted p-values from the archived `p_wilcoxon_raw` values. It does not reconstruct the per-seed paired vectors and therefore does not recompute the raw Wilcoxon tests, confidence intervals, or mean differences. Those are retained from the original B10 contrast table.
