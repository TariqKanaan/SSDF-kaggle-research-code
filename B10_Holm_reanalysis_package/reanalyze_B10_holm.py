#!/usr/bin/env python3
"""Standalone Holm re-analysis for the archived B10 paired-contrast table.

The input is the original 10-column file:
    B10_paired_persistence_contrasts.csv

The script does NOT recompute paired Wilcoxon tests or confidence intervals.
It independently applies Holm's step-down adjustment to the archived raw
Wilcoxon p-values, using one 64-test family per metric.

No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

EXPECTED_METRICS = ("J_attack", "auc_attack", "ffr_clean", "pfa_clean")
EXPECTED_PIPELINES = (
    "BDFHistory",
    "BetaHistory",
    "Huber",
    "NoDefense",
    "TTF-CF",
    "XS-PH",
    "XS-fixed20",
    "XS-fixed50",
)
EXPECTED_RHO = (0.2, 0.4)
EXPECTED_SNR = (-9.0, -6.0)
EXPECTED_PERSISTENCE_W = (0.5, 0.9)
EXPECTED_PAIRS = 96
EXPECTED_FAMILY_N = (
    len(EXPECTED_PIPELINES)
    * len(EXPECTED_RHO)
    * len(EXPECTED_SNR)
    * len(EXPECTED_PERSISTENCE_W)
)
EXPECTED_TOTAL_ROWS = len(EXPECTED_METRICS) * EXPECTED_FAMILY_N
ALPHA = 0.05

REQUIRED_COLUMNS = (
    "snr_db",
    "rho_requested",
    "pipeline",
    "persistence_w_vs_zero",
    "metric",
    "n_pairs",
    "mean_difference",
    "ci95_low",
    "ci95_high",
    "p_wilcoxon_raw",
)
OUTPUT_COLUMNS = REQUIRED_COLUMNS + (
    "holm_family_n",
    "p_holm_within_metric_family",
)


def parse_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Row {row_number}: field {field!r} is not numeric: {value!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(
            f"Row {row_number}: field {field!r} must be finite, got {value!r}"
        )
    return result


def close_to(a: float, b: float, tol: float = 1e-10) -> bool:
    return math.isclose(a, b, rel_tol=0.0, abs_tol=tol)


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    """Return Holm-adjusted p-values in their original input order."""
    m = len(p_values)
    if m == 0:
        return []
    for p in p_values:
        if not math.isfinite(p) or p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid raw p-value for Holm adjustment: {p!r}")

    order = sorted(range(m), key=lambda i: (p_values[i], i))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (m - rank) * p_values[original_index])
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max
    return adjusted


def read_and_validate(input_path: Path) -> List[Dict[str, str]]:
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("Input CSV has no header row.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Input CSV is missing required columns: {missing}")
        derived = {"holm_family_n", "p_holm_within_metric_family"}
        if derived.intersection(reader.fieldnames):
            raise ValueError(
                "Input appears to already contain Holm output columns. "
                "Use the original 10-column B10_paired_persistence_contrasts.csv "
                "to avoid silently adjusting already-adjusted values."
            )
        rows = []
        for row in reader:
            rows.append({c: row[c] for c in REQUIRED_COLUMNS})

    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_TOTAL_ROWS} rows (4 metrics x 64 tests), "
            f"but found {len(rows)}."
        )

    seen = set()
    row_counts = Counter()
    metric_keys = defaultdict(set)
    for i, row in enumerate(rows, start=2):
        snr = parse_float(row["snr_db"], "snr_db", i)
        rho = parse_float(row["rho_requested"], "rho_requested", i)
        w = parse_float(row["persistence_w_vs_zero"], "persistence_w_vs_zero", i)
        n_pairs = parse_float(row["n_pairs"], "n_pairs", i)
        p_raw = parse_float(row["p_wilcoxon_raw"], "p_wilcoxon_raw", i)

        metric = row["metric"]
        pipeline = row["pipeline"]
        if metric not in EXPECTED_METRICS:
            raise ValueError(f"Row {i}: unexpected metric {metric!r}.")
        if pipeline not in EXPECTED_PIPELINES:
            raise ValueError(f"Row {i}: unexpected pipeline {pipeline!r}.")
        if not any(close_to(rho, v) for v in EXPECTED_RHO):
            raise ValueError(f"Row {i}: unexpected rho_requested={rho}.")
        if not any(close_to(snr, v) for v in EXPECTED_SNR):
            raise ValueError(f"Row {i}: unexpected snr_db={snr}.")
        if not any(close_to(w, v) for v in EXPECTED_PERSISTENCE_W):
            raise ValueError(
                f"Row {i}: unexpected persistence_w_vs_zero={w}; "
                "only 0.5 and 0.9 contrasts against w=0 are expected."
            )
        if not close_to(n_pairs, EXPECTED_PAIRS):
            raise ValueError(
                f"Row {i}: expected {EXPECTED_PAIRS} paired seeds, got {n_pairs}."
            )
        if p_raw < 0.0 or p_raw > 1.0:
            raise ValueError(f"Row {i}: raw p-value must be within [0, 1].")

        key = (metric, pipeline, rho, snr, w)
        if key in seen:
            raise ValueError(f"Duplicate B10 contrast row: {key}")
        seen.add(key)
        metric_keys[metric].add((pipeline, rho, snr, w))
        row_counts[metric] += 1

    expected_grid = set(itertools.product(
        EXPECTED_PIPELINES,
        EXPECTED_RHO,
        EXPECTED_SNR,
        EXPECTED_PERSISTENCE_W,
    ))
    for metric in EXPECTED_METRICS:
        if row_counts[metric] != EXPECTED_FAMILY_N:
            raise ValueError(
                f"Metric {metric}: expected {EXPECTED_FAMILY_N} tests, "
                f"found {row_counts[metric]}."
            )
        if metric_keys[metric] != expected_grid:
            missing = sorted(expected_grid - metric_keys[metric])
            extra = sorted(metric_keys[metric] - expected_grid)
            raise ValueError(
                f"Metric {metric}: incomplete or unexpected 64-test grid. "
                f"Missing={missing}; extra={extra}"
            )

    return rows


def apply_holm(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_metric: Dict[str, List[int]] = defaultdict(list)
    raw_p_by_metric: Dict[str, List[float]] = defaultdict(list)
    for i, row in enumerate(rows):
        metric = row["metric"]
        by_metric[metric].append(i)
        raw_p_by_metric[metric].append(float(row["p_wilcoxon_raw"]))

    output = [dict(row) for row in rows]
    for metric in EXPECTED_METRICS:
        adjusted = holm_adjust(raw_p_by_metric[metric])
        if len(adjusted) != EXPECTED_FAMILY_N:
            raise AssertionError("Unexpected Holm family length after validation.")
        for row_index, p_adj in zip(by_metric[metric], adjusted):
            output[row_index]["holm_family_n"] = str(EXPECTED_FAMILY_N)
            output[row_index]["p_holm_within_metric_family"] = format(p_adj, ".17g")
    return output


def write_csv(path: Path, rows: Iterable[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def build_notes(rows: List[Dict[str, str]], output_path: Path) -> str:
    xsph_counts = {}
    for metric in EXPECTED_METRICS:
        selected = [
            r for r in rows
            if r["pipeline"] == "XS-PH" and r["metric"] == metric
        ]
        xsph_counts[metric] = sum(
            float(r["p_holm_within_metric_family"]) < ALPHA for r in selected
        )

    return f"""# B10 persistence analysis: standalone Holm re-analysis

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
significance level is alpha = {ALPHA:.2f}.

## Paired design and input fields

Each row is a contrast between one non-zero persistence setting (w = 0.5 or
w = 0.9) and the zero-persistence reference (w = 0), for the same pipeline,
rho, SNR, and seed. Each contrast contains {EXPECTED_PAIRS} matched test-seed
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
| `J_attack` | {xsph_counts['J_attack']} / 8 |
| `ffr_clean` | {xsph_counts['ffr_clean']} / 8 |
| `auc_attack` | {xsph_counts['auc_attack']} / 8 |
| `pfa_clean` | {xsph_counts['pfa_clean']} / 8 |

Significance is defined as the Holm-adjusted p-value being below {ALPHA:.2f}.
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
python B10_Holm_reanalysis_package/reanalyze_B10_holm.py \\
  --input analysis/B10_paired_persistence_contrasts.csv \\
  --output analysis/B10_paired_persistence_contrasts_Holm_verified.csv \\
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
"""


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Apply metric-wise Holm correction to the B10 paired contrasts."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "B10_paired_persistence_contrasts.csv",
        help="Original 10-column B10 paired-contrast CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "B10_paired_persistence_contrasts_Holm_verified.csv",
        help="Destination for the 12-column Holm-augmented CSV.",
    )
    parser.add_argument(
        "--notes",
        type=Path,
        default=here / "B10_Holm_reanalysis_notes.md",
        help="Destination for the Markdown audit note.",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    notes_path = args.notes.resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")
    if input_path == output_path:
        parser.error("Input and output paths must be different.")

    try:
        source_rows = read_and_validate(input_path)
        adjusted_rows = apply_holm(source_rows)
        write_csv(output_path, adjusted_rows)
        notes_text = build_notes(adjusted_rows, notes_path)
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        notes_path.write_text(notes_text, encoding="utf-8")
    except (OSError, ValueError) as exc:
        raise SystemExit(f"B10 Holm re-analysis failed: {exc}") from exc

    print(f"Input rows: {len(source_rows)}")
    print(f"Metric-specific Holm family size: {EXPECTED_FAMILY_N}")
    print(f"Number of metric families: {len(EXPECTED_METRICS)}")
    print(f"Total raw tests: {EXPECTED_TOTAL_ROWS}")
    print(f"Output CSV: {output_path}")
    print(f"Analysis notes: {notes_path}")
    print("XS-PH significant contrasts at Holm-adjusted p < 0.05:")
    for metric in EXPECTED_METRICS:
        count = sum(
            r["pipeline"] == "XS-PH"
            and r["metric"] == metric
            and float(r["p_holm_within_metric_family"]) < ALPHA
            for r in adjusted_rows
        )
        print(f"  {metric}: {count}/8")


if __name__ == "__main__":
    main()
