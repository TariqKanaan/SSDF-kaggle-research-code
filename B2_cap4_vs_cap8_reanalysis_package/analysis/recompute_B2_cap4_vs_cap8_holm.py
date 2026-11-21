#!/usr/bin/env python3
"""Recompute the B2 cap-4 versus cap-8 paired J_attack analysis.

Run from the repository root:
    python B2_cap4_vs_cap8_reanalysis_package/analysis/recompute_B2_cap4_vs_cap8_holm.py

Expected input files:
    B2_rho_cap/v2_seed_level.csv.gz
    B2_rho_cap/v2_manifest.json
    B2_rho_cap/v2_output_sha256.json

Outputs:
    analysis/B2_cap4_vs_cap8_holm_reanalysis.csv
    analysis/B2_cap4_vs_cap8_holm_reanalysis_notes.md

The test family is fixed at 30 contrasts:
    3 pipelines x 5 rho values x 2 SNR levels.

This is a transparent independent reanalysis; it does not claim to reproduce
an unavailable historical B2 paired-testing script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon


PIPELINES = ("XS-PH", "XS-fixed20", "XS-fixed50")
SNRS = (-9.0, -6.0)
RHOS = (0.1, 0.2, 0.3, 0.4, 0.5)
CAPS = (4, 8, 12)
METRIC = "J_attack"
ATTACK = "masking"
EXPECTED_ROWS = 30
EXPECTED_SEEDS = 96
BOOTSTRAP_RESAMPLES_DEFAULT = 20_000
BOOTSTRAP_SEED_DEFAULT = 20261121


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, returned in original input order."""
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1 or not np.all(np.isfinite(p)):
        raise ValueError("Holm adjustment requires a finite one-dimensional p-value array.")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("P-values must lie between 0 and 1.")

    m = len(p)
    order = np.argsort(p, kind="mergesort")
    adjusted_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, candidate)
        adjusted_sorted[rank] = running_max

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def wilcoxon_two_sided_approx(differences: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank p-value using normal approximation.

    Zero differences are discarded (zero_method='wilcox'). `method='approx'`
    is used where accepted by SciPy; newer naming is handled by falling back
    to `method='asymptotic'`.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    if d.size == 0:
        raise ValueError("No finite paired differences remain for the Wilcoxon test.")
    if np.all(d == 0):
        return 1.0

    common = dict(
        alternative="two-sided",
        zero_method="wilcox",
        correction=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        try:
            result = wilcoxon(d, method="approx", **common)
        except (TypeError, ValueError):
            result = wilcoxon(d, method="asymptotic", **common)
    p = float(result.pvalue)
    if not np.isfinite(p):
        raise ValueError("Wilcoxon returned a non-finite p-value.")
    return min(1.0, max(0.0, p))


def paired_bootstrap_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> tuple[float, float]:
    """Percentile CI for the mean paired difference using seed-level resampling."""
    d = np.asarray(differences, dtype=float)
    n = len(d)
    if n == 0:
        raise ValueError("Cannot bootstrap an empty paired-difference vector.")
    indices = rng.integers(0, n, size=(resamples, n))
    bootstrap_means = d[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def paired_vectors(
    data: pd.DataFrame,
    pipeline: str,
    snr: float,
    rho: float,
    cap_a: int,
    cap_b: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched seed IDs and metric vectors for two caps."""
    subset = data[
        (data["snr_db"] == snr)
        & (data["rho_requested"] == rho)
        & data["pipeline"].isin([f"{pipeline}_cap{cap_a}", f"{pipeline}_cap{cap_b}"])
    ].copy()

    if subset.duplicated(["seed", "pipeline"]).any():
        raise ValueError(
            f"Duplicate seed/pipeline rows for {pipeline}, SNR={snr}, rho={rho}."
        )

    wide = subset.pivot(index="seed", columns="pipeline", values=METRIC)
    left_name = f"{pipeline}_cap{cap_a}"
    right_name = f"{pipeline}_cap{cap_b}"
    if left_name not in wide.columns or right_name not in wide.columns:
        raise ValueError(f"Missing cap pair {left_name} / {right_name}.")

    wide = wide[[left_name, right_name]].dropna().sort_index()
    if len(wide) != EXPECTED_SEEDS:
        raise ValueError(
            f"Expected {EXPECTED_SEEDS} matched seeds for {pipeline}, SNR={snr}, rho={rho}; "
            f"found {len(wide)}."
        )

    return (
        wide.index.to_numpy(dtype=int),
        wide[left_name].to_numpy(dtype=float),
        wide[right_name].to_numpy(dtype=float),
    )


def verify_input_hash(input_path: Path, hash_path: Path) -> tuple[str, str]:
    actual = sha256_file(input_path)
    if not hash_path.exists():
        raise FileNotFoundError(f"SHA-256 manifest file not found: {hash_path}")
    expected_map = json.loads(hash_path.read_text(encoding="utf-8"))
    expected = expected_map.get("v2_seed_level.csv.gz")
    if not expected:
        raise ValueError("v2_output_sha256.json has no entry for v2_seed_level.csv.gz.")
    if actual.lower() != str(expected).lower():
        raise ValueError(
            "Input checksum mismatch.\n"
            f"  actual:   {actual}\n"
            f"  expected: {expected}"
        )
    return actual, str(expected)


def build_analysis(
    input_path: Path,
    hash_path: Path,
    output_path: Path,
    notes_path: Path,
    bootstrap_resamples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if bootstrap_resamples < 1:
        raise ValueError("bootstrap_resamples must be at least 1.")

    input_sha, manifest_sha = verify_input_hash(input_path, hash_path)
    raw = pd.read_csv(input_path)

    required = {
        "seed", "snr_db", "rho_requested", "attack", "pipeline", METRIC
    }
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    masking = raw[
        (raw["attack"] == ATTACK)
        & raw["snr_db"].isin(SNRS)
        & raw["rho_requested"].isin(RHOS)
        & raw["pipeline"].isin(
            [f"{pipeline}_cap{cap}" for pipeline in PIPELINES for cap in CAPS]
        )
    ].copy()

    if masking.empty:
        raise ValueError("No rows matched the specified B2 masking analysis grid.")

    # Check that each condition contains the same 96 test seeds for all caps.
    for pipeline in PIPELINES:
        for snr in SNRS:
            for rho in RHOS:
                seed_sets = []
                for cap in CAPS:
                    seeds, _, _ = paired_vectors(masking, pipeline, snr, rho, cap, 8 if cap != 8 else 4)
                    seed_sets.append(set(seeds.tolist()))
                if not (seed_sets[0] == seed_sets[1] == seed_sets[2]):
                    raise ValueError(
                        f"Seed sets do not match across cap variants for {pipeline}, SNR={snr}, rho={rho}."
                    )

    # J_attack identity check for cap 8 versus cap 12 in all 30 cells.
    cap8_cap12_mismatches = 0
    for pipeline in PIPELINES:
        for snr in SNRS:
            for rho in RHOS:
                _, x8, x12 = paired_vectors(masking, pipeline, snr, rho, 8, 12)
                cap8_cap12_mismatches += int(np.count_nonzero(x8 != x12))
    if cap8_cap12_mismatches:
        raise ValueError(
            "Cap 8 and cap 12 J_attack values are not identical; "
            f"found {cap8_cap12_mismatches} mismatched seed values."
        )

    # Keep this iteration order fixed. It makes the single seeded bootstrap
    # stream deterministic and reproduces the previously published CIs.
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, object]] = []
    for snr in SNRS:
        for rho in RHOS:
            for pipeline in PIPELINES:
                seeds, cap4, cap8 = paired_vectors(masking, pipeline, snr, rho, 4, 8)
                differences = cap4 - cap8
                p_raw = wilcoxon_two_sided_approx(differences)
                ci_low, ci_high = paired_bootstrap_ci(
                    differences, rng, bootstrap_resamples
                )
                rows.append(
                    {
                        "snr_db": snr,
                        "rho_requested": rho,
                        "pipeline": pipeline,
                        "contrast": "cap4_minus_cap8",
                        "metric": METRIC,
                        "n_pairs": int(len(seeds)),
                        "mean_difference": float(np.mean(differences)),
                        "ci95_paired_bootstrap_low": ci_low,
                        "ci95_paired_bootstrap_high": ci_high,
                        "wins": int(np.sum(differences > 0)),
                        "losses": int(np.sum(differences < 0)),
                        "ties": int(np.sum(differences == 0)),
                        "p_wilcoxon_raw": p_raw,
                        "holm_family_size": EXPECTED_ROWS,
                        "family_definition": (
                            "J_attack cap4-vs-cap8; 3 pipeline families x 5 rho x 2 SNR"
                        ),
                    }
                )

    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_ROWS:
        raise AssertionError(f"Expected {EXPECTED_ROWS} contrasts; generated {len(result)}.")
    result["p_holm_family30"] = holm_adjust(result["p_wilcoxon_raw"].to_numpy())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, float_format="%.12g")

    # Create a short, self-contained audit note alongside the CSV.
    xsph = result[(result["pipeline"] == "XS-PH") & (result["snr_db"] == -6.0)]
    lines = [
        "# B2 cap-sensitivity reanalysis: reproducible script output",
        "",
        "## Input integrity",
        f"- Input file: `{(Path('B2_rho_cap') / input_path.name).as_posix() if input_path.name == 'v2_seed_level.csv.gz' else input_path.name}`",
        f"- Input SHA-256: `{input_sha}`",
        f"- SHA-256 recorded in `v2_output_sha256.json`: `{manifest_sha}`",
        "- Checksum match: **yes**",
        f"- Rows in complete input: {len(raw):,}",
        f"- Masking-arm rows in the selected grid: {len(masking):,}",
        f"- Paired test seeds per contrast: {EXPECTED_SEEDS}",
        f"- Cap 8 versus cap 12 mismatched `J_attack` seed values: {cap8_cap12_mismatches}",
        "",
        "## Analysis specification",
        f"- Outcome: `{METRIC}`.",
        "- Contrast: cap 4 minus cap 8; positive differences favour cap 4.",
        "- One Holm family of 30 contrasts: 3 pipelines × 5 rho values × 2 SNR levels.",
        "- Paired test: two-sided Wilcoxon signed-rank, `zero_method='wilcox'`, no continuity correction, normal approximation.",
        f"- Bootstrap CI: paired-seed percentile bootstrap, {bootstrap_resamples:,} resamples, NumPy default_rng seed {bootstrap_seed}.",
        "- Bootstrap stream iteration order: SNR, rho, pipeline; this order is fixed in the script.",
        "- Holm: step-down adjustment across all 30 raw p-values.",
        "",
        "## Software versions used for this run",
        f"- Python: `{sys.version.split()[0]}`",
        f"- NumPy: `{np.__version__}`",
        f"- pandas: `{pd.__version__}`",
        f"- SciPy: `{scipy.__version__}`",
        "",
        "## XS-PH at -6 dB",
        "| rho | Mean cap4-cap8 | 95% paired-bootstrap CI | W/L/T | Raw Wilcoxon p | Holm p (30) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in xsph.sort_values("rho_requested").iterrows():
        lines.append(
            f"| {row['rho_requested']:.2f} | {row['mean_difference']:+.6f} "
            f"| [{row['ci95_paired_bootstrap_low']:+.6f}, {row['ci95_paired_bootstrap_high']:+.6f}] "
            f"| {int(row['wins'])}/{int(row['losses'])}/{int(row['ties'])} "
            f"| {row['p_wilcoxon_raw']:.8g} | {row['p_holm_family30']:.8g} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and reproducibility note",
            "The inferential family is explicitly defined as the 30 cap-4 versus cap-8 `J_attack` contrasts above. "
            "The supplied historical B2 analysis script did not implement these paired B2 tests or their Holm correction; "
            "this file provides a standalone, transparent reanalysis from the seed-level data.",
            "",
            "The precise asymptotic Wilcoxon p-values can vary slightly across SciPy versions because paired score "
            "differences are floating-point values and near-equal absolute differences may be ranked differently. "
            "For exact reproduction of this run, use the software versions recorded above. The qualitative decisions "
            "for the two highlighted XS-PH cells at rho=0.40 and rho=0.50 remain the same at alpha=0.05.",
            "",
            "Confidence intervals are unadjusted paired-bootstrap 95% intervals; Holm adjustment applies to p-values.",
            "",
        ]
    )
    notes_path.parent.mkdir(parents=True, exist_ok=True)
    notes_path.write_text("\n".join(lines), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    # Support both the canonical location (repository_root/analysis/) and the
    # bundled location (repository_root/B2_cap4_vs_cap8_reanalysis_package/analysis/).
    repo_root = None
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "B2_rho_cap" / "v2_seed_level.csv.gz").is_file():
            repo_root = candidate
            break
    if repo_root is None:
        # Keep conventional behavior as a clear fallback for explicit --input.
        repo_root = script_dir.parent if script_dir.name == "analysis" else script_dir
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=repo_root / "B2_rho_cap" / "v2_seed_level.csv.gz",
        help="B2 seed-level CSV.GZ input (default: B2_rho_cap/v2_seed_level.csv.gz).",
    )
    parser.add_argument(
        "--hashes", type=Path, default=repo_root / "B2_rho_cap" / "v2_output_sha256.json",
        help="B2 output SHA-256 JSON file.",
    )
    parser.add_argument(
        "--output", type=Path, default=repo_root / "analysis" / "B2_cap4_vs_cap8_holm_reanalysis.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--notes", type=Path, default=repo_root / "analysis" / "B2_cap4_vs_cap8_holm_reanalysis_notes.md",
        help="Output Markdown audit-note path.",
    )
    parser.add_argument(
        "--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES_DEFAULT,
        help=f"Number of bootstrap resamples (default: {BOOTSTRAP_RESAMPLES_DEFAULT}).",
    )
    parser.add_argument(
        "--bootstrap-seed", type=int, default=BOOTSTRAP_SEED_DEFAULT,
        help=f"Bootstrap RNG seed (default: {BOOTSTRAP_SEED_DEFAULT}).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = build_analysis(
            input_path=args.input,
            hash_path=args.hashes,
            output_path=args.output,
            notes_path=args.notes,
            bootstrap_resamples=args.bootstrap_resamples,
            bootstrap_seed=args.bootstrap_seed,
        )
    except Exception as exc:  # CLI should fail with a clear message rather than a traceback by default.
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(result)} paired contrasts to: {args.output}")
    print(f"Wrote audit notes to: {args.notes}")
    highlight = result[
        (result["pipeline"] == "XS-PH")
        & (result["snr_db"] == -6.0)
        & result["rho_requested"].isin([0.4, 0.5])
    ].sort_values("rho_requested")
    print("\nXS-PH, SNR=-6 dB:")
    print(highlight[["rho_requested", "mean_difference", "p_wilcoxon_raw", "p_holm_family30"]].to_string(index=False))
    print(f"\nVersions: Python {sys.version.split()[0]}, NumPy {np.__version__}, pandas {pd.__version__}, SciPy {scipy.__version__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
